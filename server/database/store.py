"""
AgentVault — DatabaseStore (Unified Facade)
Same public API as the old monolithic class — delegates to the active provider.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import DatabaseProvider
from .factory import create_provider

logger = logging.getLogger("agentvault.database.store")


class DatabaseStore:
    """
    Drop-in replacement for the old ``DatabaseStore``.

    All methods mirror the original API so that ``routes.py`` and ``app.py``
    work without any changes beyond the import path.  Internally, every call
    is forwarded to whichever ``DatabaseProvider`` was created by the factory.
    """

    def __init__(self, uri: str = "agentvault.db") -> None:
        self._uri = uri
        self._provider: Optional[DatabaseProvider] = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the provider, connect, and health-check in one call."""
        self._provider = await create_provider(self._uri)
        logger.info("DatabaseStore ready (backend=%s)", self._uri.split("://")[0] if "://" in self._uri else "sqlite")

    async def close(self) -> None:
        if self._provider:
            await self._provider.close()

    async def health_check(self) -> dict[str, Any]:
        return await self._p.health_check()

    @property
    def _p(self) -> DatabaseProvider:
        if not self._provider:
            raise RuntimeError("DatabaseStore not connected — call connect() first")
        return self._provider

    # ── audit events ─────────────────────────────────────────────────

    async def save_audit_event(self, event: dict) -> None:
        await self._p.save_audit_event(event)

    async def query_audit(
        self,
        agent_id: Optional[str] = None,
        decision: Optional[str] = None,
        action_name: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        return await self._p.query_audit(
            agent_id=agent_id, decision=decision, action_name=action_name,
            session_id=session_id, start_time=start_time, end_time=end_time,
            limit=limit, offset=offset,
        )

    async def get_session_replay(self, session_id: str) -> list[dict]:
        return await self._p.get_session_replay(session_id)

    async def verify_chain(self) -> dict:
        return await self._p.verify_chain()

    # ── drift alerts ─────────────────────────────────────────────────

    async def save_drift_alert(self, alert: dict) -> None:
        await self._p.save_drift_alert(alert)

    async def query_drift_alerts(
        self,
        agent_id: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        return await self._p.query_drift_alerts(agent_id=agent_id, level=level, limit=limit)

    # ── escalation queue ─────────────────────────────────────────────

    async def save_escalation(self, escalation: dict) -> None:
        await self._p.save_escalation(escalation)

    async def get_escalations(self, status: Optional[str] = None) -> list[dict]:
        return await self._p.get_escalations(status)

    async def resolve_escalation(
        self, escalation_id: str, approved: bool, resolved_by: str = "admin"
    ) -> Optional[dict]:
        return await self._p.resolve_escalation(escalation_id, approved, resolved_by)

    # ── agents ───────────────────────────────────────────────────────

    async def get_agents(self) -> list[dict]:
        return await self._p.get_agents()

    # ── stats ────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        return await self._p.get_stats()

    # ── users ────────────────────────────────────────────────────────

    async def create_user(self, user: dict) -> None:
        await self._p.create_user(user)

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        return await self._p.get_user_by_email(email)

    # ── API keys ─────────────────────────────────────────────────────

    async def save_api_key(self, key_data: dict) -> None:
        await self._p.save_api_key(key_data)

    async def get_api_keys(self, user_id: str) -> list[dict]:
        return await self._p.get_api_keys(user_id)

    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        return await self._p.revoke_api_key(key_id, user_id)

    async def verify_api_key(self, raw_key: str) -> Optional[dict]:
        return await self._p.verify_api_key(raw_key)
