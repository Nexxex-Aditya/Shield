"""
AgentVault — Tool Chain Analyzer

Detects dangerous action sequences that span multiple tool calls.
Individual calls may be safe, but the chain reveals intent:
    read_credentials → encode_base64 → send_email = data exfiltration

Also supports multi-agent collective policy — detects coordinated
attacks where Agent A reads and Agent B sends.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from collections import defaultdict
from typing import Optional

from .models import ChainViolation

logger = logging.getLogger("agentvault.chain_analyzer")


# Default forbidden sequences
DEFAULT_FORBIDDEN_SEQUENCES = [
    {
        "pattern": ["read_*", "send_*"],
        "reasoning": "Potential data exfiltration: read then send",
        "window": 300,  # 5 minute window
    },
    {
        "pattern": ["read_*", "api_call"],
        "reasoning": "Potential data exfiltration via API",
        "window": 300,
    },
    {
        "pattern": ["get_*_key*", "*"],
        "reasoning": "Credential access followed by any action is suspicious",
        "window": 120,
    },
    {
        "pattern": ["read_*", "write_*", "delete_*"],
        "reasoning": "Read-modify-delete pattern detected",
        "window": 600,
    },
    {
        "pattern": ["list_*", "read_*", "read_*", "read_*"],
        "reasoning": "Bulk data harvesting: listing then mass reading",
        "window": 120,
    },
    {
        "pattern": ["execute_code", "send_*"],
        "reasoning": "Code execution followed by data send",
        "window": 300,
    },
    {
        "pattern": ["modify_permissions", "*"],
        "reasoning": "Permission escalation followed by action",
        "window": 600,
    },
]


class ChainAnalyzer:
    """
    Analyzes sequences of tool calls to detect dangerous patterns.

    Two modes:
    1. Per-session: tracks action chains within a single agent session
    2. Collective: tracks cross-agent patterns (Agent A reads, Agent B sends)

    Uses glob-pattern matching for flexible sequence definitions.
    Sliding window ensures sequences are only matched within a time frame.
    """

    def __init__(self, max_history: int = 200) -> None:
        self._forbidden: list[dict] = []
        self._session_history: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
        # For collective: global recent actions across all agents
        self._global_history: list[tuple[float, str, str, str]] = []  # (time, agent, session, action)
        self._violations: list[ChainViolation] = []
        self._max_history = max_history
        self._lock = threading.Lock()

    def load_defaults(self) -> None:
        """Load default forbidden sequences."""
        for seq in DEFAULT_FORBIDDEN_SEQUENCES:
            self.add_forbidden_sequence(**seq)

    def add_forbidden_sequence(
        self,
        pattern: list[str],
        reasoning: str = "",
        window: int = 300,
    ) -> None:
        """Add a forbidden action sequence pattern."""
        self._forbidden.append({
            "pattern": pattern,
            "reasoning": reasoning,
            "window": window,
        })

    def record_and_check(
        self,
        agent_id: str,
        session_id: str,
        action_name: str,
    ) -> Optional[ChainViolation]:
        """
        Record an action and check for forbidden sequences.
        Returns a ChainViolation if a pattern is matched.
        """
        now = time.time()

        with self._lock:
            # Record in session history
            key = f"{agent_id}:{session_id}"
            self._session_history[key].append((now, action_name, agent_id))

            # Trim to max history
            if len(self._session_history[key]) > self._max_history:
                self._session_history[key] = self._session_history[key][-self._max_history:]

            # Record in global history (for collective detection)
            self._global_history.append((now, agent_id, session_id, action_name))
            if len(self._global_history) > self._max_history * 5:
                self._global_history = self._global_history[-self._max_history * 5:]

        # Check session-level sequences
        violation = self._check_sequences(
            agent_id, session_id,
            [(t, a) for t, a, _ in self._session_history.get(key, [])],
        )
        if violation:
            return violation

        # Check collective sequences (cross-agent)
        violation = self._check_collective(agent_id, session_id)
        return violation

    def _check_sequences(
        self,
        agent_id: str,
        session_id: str,
        history: list[tuple[float, str]],
    ) -> Optional[ChainViolation]:
        """Check a history of (timestamp, action) against forbidden patterns."""
        for rule in self._forbidden:
            pattern = rule["pattern"]
            window = rule["window"]

            if len(history) < len(pattern):
                continue

            # Sliding window: check the most recent N actions within the time window
            now = time.time()
            recent = [(t, a) for t, a in history if now - t <= window]

            if len(recent) < len(pattern):
                continue

            # Try to match the pattern within recent actions
            match = self._match_pattern(pattern, [a for _, a in recent])
            if match:
                violation = ChainViolation(
                    agent_id=agent_id,
                    session_id=session_id,
                    matched_pattern=pattern,
                    actual_sequence=match,
                    reasoning=rule["reasoning"],
                )
                with self._lock:
                    self._violations.append(violation)

                logger.warning(
                    "🔗 CHAIN VIOLATION: Agent '%s' matched pattern %s → %s",
                    agent_id, pattern, match,
                )
                return violation

        return None

    def _check_collective(
        self, agent_id: str, session_id: str
    ) -> Optional[ChainViolation]:
        """Check for cross-agent coordinated patterns."""
        now = time.time()

        for rule in self._forbidden:
            pattern = rule["pattern"]
            window = rule["window"]

            # Get recent global actions within window
            recent = [
                (t, ag, sid, action)
                for t, ag, sid, action in self._global_history
                if now - t <= window
            ]

            if len(recent) < len(pattern):
                continue

            # Check if different agents collectively form a forbidden pattern
            actions = [action for _, _, _, action in recent]
            agents_in_window = set(ag for _, ag, _, _ in recent)

            if len(agents_in_window) < 2:
                continue  # Same agent — already covered by session check

            match = self._match_pattern(pattern, actions)
            if match:
                violation = ChainViolation(
                    agent_id=agent_id,
                    session_id=session_id,
                    matched_pattern=pattern,
                    actual_sequence=match,
                    reasoning=f"COLLECTIVE: {rule['reasoning']} (multi-agent coordinated)",
                )
                with self._lock:
                    self._violations.append(violation)

                logger.warning(
                    "🔗 COLLECTIVE CHAIN VIOLATION: Agents %s matched pattern %s",
                    agents_in_window, pattern,
                )
                return violation

        return None

    def _match_pattern(
        self, pattern: list[str], actions: list[str]
    ) -> Optional[list[str]]:
        """
        Check if a pattern exists as a subsequence in actions (glob matching).
        Returns the matched actual actions, or None.
        """
        matched = []
        pattern_idx = 0

        for action in actions:
            if pattern_idx >= len(pattern):
                break
            if fnmatch.fnmatch(action, pattern[pattern_idx]):
                matched.append(action)
                pattern_idx += 1

        if pattern_idx == len(pattern):
            return matched
        return None

    def get_violations(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[ChainViolation]:
        """Get chain violation history."""
        with self._lock:
            violations = list(self._violations)
        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]
        return sorted(violations, key=lambda v: v.timestamp, reverse=True)[:limit]

    def get_session_chain(self, agent_id: str, session_id: str) -> list[str]:
        """Get the action chain for a specific session."""
        key = f"{agent_id}:{session_id}"
        return [action for _, action, _ in self._session_history.get(key, [])]

    def clear(self) -> None:
        """Clear all data (testing only)."""
        with self._lock:
            self._session_history.clear()
            self._global_history.clear()
            self._violations.clear()
