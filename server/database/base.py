"""
AgentVault — Database Provider Interface
Abstract base class that every backend must implement.
"""

from __future__ import annotations

import abc
from typing import Any, Optional


class DatabaseProvider(abc.ABC):
    """
    13-method contract for all database backends.

    Every provider exposes the same surface — the ``DatabaseStore`` facade
    delegates transparently to whichever provider is active, so the rest of
    the application never touches driver-specific code.
    """

    # ── lifecycle ────────────────────────────────────────────────────

    @abc.abstractmethod
    async def connect(self) -> None:
        """Initialise connection pool / session and create schema if needed."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Tear down connections gracefully."""

    @abc.abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Return a health probe dict, e.g.
        {"ok": True, "backend": "postgresql", "latency_ms": 3.2}
        """

    # ── audit events ─────────────────────────────────────────────────

    @abc.abstractmethod
    async def save_audit_event(self, event: dict) -> None: ...

    @abc.abstractmethod
    async def query_audit(
        self,
        *,
        agent_id: Optional[str] = None,
        decision: Optional[str] = None,
        action_name: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]: ...

    @abc.abstractmethod
    async def get_session_replay(self, session_id: str) -> list[dict]: ...

    @abc.abstractmethod
    async def verify_chain(self) -> dict: ...

    # ── drift alerts ─────────────────────────────────────────────────

    @abc.abstractmethod
    async def save_drift_alert(self, alert: dict) -> None: ...

    @abc.abstractmethod
    async def query_drift_alerts(
        self,
        *,
        agent_id: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]: ...

    # ── escalation queue ─────────────────────────────────────────────

    @abc.abstractmethod
    async def save_escalation(self, escalation: dict) -> None: ...

    @abc.abstractmethod
    async def get_escalations(self, status: Optional[str] = None) -> list[dict]: ...

    @abc.abstractmethod
    async def resolve_escalation(
        self, escalation_id: str, approved: bool, resolved_by: str = "admin"
    ) -> Optional[dict]: ...

    # ── agents ───────────────────────────────────────────────────────

    @abc.abstractmethod
    async def get_agents(self) -> list[dict]: ...

    # ── stats ────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def get_stats(self) -> dict: ...

    # ── users ────────────────────────────────────────────────────────

    @abc.abstractmethod
    async def create_user(self, user: dict) -> None: ...

    @abc.abstractmethod
    async def get_user_by_email(self, email: str) -> Optional[dict]: ...

    # ── API keys ─────────────────────────────────────────────────────

    @abc.abstractmethod
    async def save_api_key(self, key_data: dict) -> None: ...

    @abc.abstractmethod
    async def get_api_keys(self, user_id: str) -> list[dict]: ...

    @abc.abstractmethod
    async def revoke_api_key(self, key_id: str, user_id: str) -> bool: ...

    @abc.abstractmethod
    async def verify_api_key(self, raw_key: str) -> Optional[dict]: ...
