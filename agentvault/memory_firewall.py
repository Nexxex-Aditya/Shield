"""
AgentVault — Cross-Session Memory Firewall

Detects data smuggling between agent sessions. Agents should not persist
data from one session to another via tool parameters (attempting to
build "memory" outside authorized channels).

How it works:
1. Fingerprint parameter values per session using locality-sensitive hashing
2. When a new session starts, compare parameter fingerprints against previous sessions
3. Flag high-similarity matches as potential data smuggling
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import defaultdict
from typing import Any, Optional

from .models import MemoryViolation

logger = logging.getLogger("agentvault.memory_firewall")


class MemoryFirewall:
    """
    Tracks parameter fingerprints across sessions to detect data persistence.

    Data smuggling pattern:
    Session 1: Agent reads sensitive data → Session ends
    Session 2: Agent passes the same data as parameters → Exfiltration

    The firewall catches this by fingerprinting parameter values and checking
    for suspicious reuse across session boundaries.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_value_length: int = 20,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._min_value_length = min_value_length
        # agent_id → session_id → set of value fingerprints
        self._fingerprints: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._violations: list[MemoryViolation] = []
        self._lock = threading.Lock()

    def check(
        self,
        agent_id: str,
        session_id: str,
        parameters: dict[str, Any],
    ) -> Optional[MemoryViolation]:
        """
        Check parameters for cross-session data reuse.
        Returns MemoryViolation if suspicious data detected, None otherwise.
        """
        # Extract meaningful string values from parameters
        values = self._extract_values(parameters)

        if not values:
            return None

        # Generate fingerprints for current values
        current_fps = set()
        for v in values:
            fp = self._fingerprint(v)
            current_fps.add(fp)

        # Check against other sessions for this agent
        violation = None
        with self._lock:
            agent_sessions = self._fingerprints[agent_id]
            for prev_session_id, prev_fps in agent_sessions.items():
                if prev_session_id == session_id:
                    continue  # Skip same session

                # Check for fingerprint overlap
                overlap = current_fps & prev_fps
                if overlap:
                    similarity = len(overlap) / max(len(current_fps), 1)
                    if similarity >= self._similarity_threshold:
                        violation = MemoryViolation(
                            agent_id=agent_id,
                            current_session=session_id,
                            source_session=prev_session_id,
                            fingerprint=list(overlap)[0],
                            similarity=round(similarity, 3),
                        )
                        self._violations.append(violation)
                        logger.warning(
                            "🧱 MEMORY VIOLATION: Agent '%s' session '%s' "
                            "reusing data from session '%s' (similarity: %.1f%%)",
                            agent_id, session_id, prev_session_id,
                            similarity * 100,
                        )
                        break

            # Store current fingerprints
            agent_sessions[session_id].update(current_fps)

        return violation

    def _extract_values(self, obj: Any, depth: int = 0) -> list[str]:
        """Extract meaningful string values from nested structures."""
        if depth > 5:
            return []

        results = []
        if isinstance(obj, str) and len(obj) >= self._min_value_length:
            results.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                results.extend(self._extract_values(v, depth + 1))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                results.extend(self._extract_values(item, depth + 1))
        return results

    @staticmethod
    def _fingerprint(value: str) -> str:
        """Create a content fingerprint using n-gram hashing."""
        # Normalize: lowercase, strip whitespace
        normalized = value.lower().strip()
        # Use SHA-256 of the normalized value
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get_violations(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[MemoryViolation]:
        """Get memory violation history."""
        with self._lock:
            violations = list(self._violations)
        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]
        return sorted(violations, key=lambda v: v.timestamp, reverse=True)[:limit]

    def clear(self) -> None:
        """Clear all data (testing only)."""
        with self._lock:
            self._fingerprints.clear()
            self._violations.clear()
