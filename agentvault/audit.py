"""
AgentVault — Tamper-Proof Audit Chain
SHA-256 hash-chain logger with session replay, filtered queries,
chain verification, and JSON export.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime
from typing import Any, Callable, Optional

from .models import AuditEvent, Decision, LogLevel

logger = logging.getLogger("agentvault.audit")


class AuditChain:
    """
    Tamper-proof audit logger using SHA-256 hash chain.
    
    Each event's hash = SHA-256(previous_hash + event_payload).
    This creates an immutable chain — if any event is modified,
    all subsequent hashes break, and verify() detects it.
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()
        self._previous_hash = "GENESIS"
        self._listeners: list[Callable[[AuditEvent], None]] = []

    @staticmethod
    def _compute_hash(previous_hash: str, payload: str) -> str:
        """Compute SHA-256 hash of previous_hash + payload."""
        data = f"{previous_hash}:{payload}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_data(data: Any) -> str:
        """Hash arbitrary data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def log(self, event: AuditEvent) -> AuditEvent:
        """
        Append an immutable audit event to the chain.
        Computes and sets event_hash and previous_hash before storing.
        """
        with self._lock:
            # Compute input/output hashes if not set
            if not event.input_hash:
                event.input_hash = self._hash_data(event.parameters)

            if not event.output_hash and event.result:
                event.output_hash = self._hash_data(event.result)

            # Set chain hashes
            event.previous_hash = self._previous_hash

            # Create payload for hashing (exclude the hash fields themselves)
            payload_dict = event.model_dump(exclude={"event_hash", "previous_hash"})
            payload = json.dumps(payload_dict, sort_keys=True, default=str)

            event.event_hash = self._compute_hash(self._previous_hash, payload)
            self._previous_hash = event.event_hash

            self._events.append(event)

            logger.info(
                "Audit [%s] %s:%s → %s (hash: %s...)",
                event.agent_id,
                event.tool_name,
                event.action_name,
                event.decision.value,
                event.event_hash[:12],
            )

        # Notify listeners (outside lock to prevent deadlock)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error("Audit listener error: %s", e)

        return event

    def log_action(
        self,
        agent_id: str,
        session_id: str,
        trace_id: str,
        action_name: str,
        tool_name: str,
        parameters: dict,
        decision: Decision,
        reasoning: str,
        confidence_score: Optional[float] = None,
        drift_score: Optional[float] = None,
        result: Optional[dict] = None,
        log_level: LogLevel = LogLevel.STANDARD,
    ) -> AuditEvent:
        """Convenience method to create and log an audit event."""
        event = AuditEvent(
            agent_id=agent_id,
            session_id=session_id,
            trace_id=trace_id,
            action_name=action_name,
            tool_name=tool_name,
            parameters=parameters if log_level != LogLevel.MINIMAL else {},
            decision=decision,
            reasoning=reasoning,
            confidence_score=confidence_score,
            drift_score=drift_score,
            result=result if log_level == LogLevel.FULL else None,
            log_level=log_level,
        )
        return self.log(event)

    def verify(self) -> tuple[bool, Optional[int]]:
        """
        Verify the entire hash chain integrity.
        Returns (is_valid, break_index).
        If valid, break_index is None.
        If tampered, break_index is the index of the first broken event.
        """
        with self._lock:
            events = list(self._events)

        if not events:
            return True, None

        prev_hash = "GENESIS"

        for i, event in enumerate(events):
            # Verify previous_hash link
            if event.previous_hash != prev_hash:
                logger.error("Chain break at index %d: previous_hash mismatch", i)
                return False, i

            # Recompute the hash
            payload_dict = event.model_dump(exclude={"event_hash", "previous_hash"})
            payload = json.dumps(payload_dict, sort_keys=True, default=str)
            expected_hash = self._compute_hash(prev_hash, payload)

            if event.event_hash != expected_hash:
                logger.error("Chain break at index %d: event_hash mismatch", i)
                return False, i

            prev_hash = event.event_hash

        return True, None

    def get_session(self, session_id: str) -> list[AuditEvent]:
        """Get all events for a session, ordered by timestamp."""
        with self._lock:
            events = [e for e in self._events if e.session_id == session_id]
        return sorted(events, key=lambda e: e.timestamp)

    def get_sessions(self) -> list[str]:
        """Get all unique session IDs."""
        with self._lock:
            seen = set()
            sessions = []
            for e in self._events:
                if e.session_id not in seen:
                    seen.add(e.session_id)
                    sessions.append(e.session_id)
        return sessions

    def query(
        self,
        agent_id: Optional[str] = None,
        decision: Optional[Decision] = None,
        action_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query events with filters."""
        with self._lock:
            events = list(self._events)

        # Apply filters
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if decision:
            events = [e for e in events if e.decision == decision]
        if action_name:
            events = [e for e in events if e.action_name == action_name]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Sort by timestamp descending (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)

        return events[offset : offset + limit]

    def get_all(self) -> list[AuditEvent]:
        """Get all events."""
        with self._lock:
            return list(self._events)

    @property
    def count(self) -> int:
        """Total number of events."""
        with self._lock:
            return len(self._events)

    def export_json(self) -> str:
        """Export entire audit chain as JSON."""
        with self._lock:
            events = list(self._events)
        return json.dumps(
            [e.model_dump(mode="json") for e in events],
            indent=2,
            default=str,
        )

    def add_listener(self, callback: Callable[[AuditEvent], None]) -> None:
        """Add a listener that gets called for every new event."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[AuditEvent], None]) -> None:
        """Remove a listener."""
        self._listeners = [l for l in self._listeners if l != callback]

    def get_stats(self) -> dict[str, int]:
        """Get aggregate statistics."""
        with self._lock:
            events = list(self._events)

        stats = {
            "total": len(events),
            "allowed": 0,
            "denied": 0,
            "escalated": 0,
        }
        for e in events:
            if e.decision == Decision.ALLOW:
                stats["allowed"] += 1
            elif e.decision == Decision.DENY:
                stats["denied"] += 1
            elif e.decision == Decision.ESCALATE:
                stats["escalated"] += 1

        return stats

    def get_agent_stats(self, agent_id: str) -> dict:
        """Get statistics for a specific agent."""
        with self._lock:
            events = [e for e in self._events if e.agent_id == agent_id]

        action_counts: dict[str, int] = {}
        decisions = {"allowed": 0, "denied": 0, "escalated": 0}

        for e in events:
            action_counts[e.action_name] = action_counts.get(e.action_name, 0) + 1
            if e.decision == Decision.ALLOW:
                decisions["allowed"] += 1
            elif e.decision == Decision.DENY:
                decisions["denied"] += 1
            elif e.decision == Decision.ESCALATE:
                decisions["escalated"] += 1

        return {
            "agent_id": agent_id,
            "total": len(events),
            "action_distribution": action_counts,
            "last_active": events[-1].timestamp.isoformat() if events else None,
            **decisions,
        }

    def clear(self) -> None:
        """Clear all events (for testing only)."""
        with self._lock:
            self._events.clear()
            self._previous_hash = "GENESIS"
