"""
AgentVault — MongoDB Provider
Document-native backend via motor (async MongoDB driver).

Supports: MongoDB Atlas, AWS DocumentDB, Azure CosmosDB (Mongo API), self-hosted.

URI formats:
    mongodb://user:pass@host:27017/dbname
    mongodb+srv://user:pass@cluster.mongodb.net/dbname
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Optional

from ..base import DatabaseProvider

logger = logging.getLogger("agentvault.db.mongo")


class MongoProvider(DatabaseProvider):
    """
    MongoDB backend using motor async driver.

    Install: ``pip install motor``
    """

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._client = None
        self._db = None

    async def connect(self) -> None:
        try:
            import motor.motor_asyncio as motor
        except ImportError:
            raise ImportError(
                "motor is required for MongoDB support. "
                "Install it: pip install motor"
            )
        from urllib.parse import urlparse
        parsed = urlparse(self._uri)
        db_name = parsed.path.lstrip("/") or "agentvault"

        self._client = motor.AsyncIOMotorClient(self._uri)
        self._db = self._client[db_name]

        # Create indexes
        await self._db.audit_events.create_index("agent_id")
        await self._db.audit_events.create_index("session_id")
        await self._db.audit_events.create_index("timestamp")
        await self._db.audit_events.create_index("decision")
        await self._db.audit_events.create_index("action_name")
        await self._db.drift_alerts.create_index("agent_id")
        await self._db.drift_alerts.create_index("alert_level")
        await self._db.escalation_queue.create_index("status")
        logger.info("MongoDB connected to %s (db=%s)", parsed.hostname, db_name)

    async def close(self) -> None:
        if self._client:
            self._client.close()

    async def health_check(self) -> dict[str, Any]:
        t0 = time.monotonic()
        info = await self._client.server_info()
        return {
            "ok": True, "backend": "mongodb",
            "latency_ms": round((time.monotonic() - t0) * 1000, 2),
            "version": info.get("version", "?"),
        }

    @property
    def db(self):
        if not self._db:
            raise RuntimeError("Not connected")
        return self._db

    @staticmethod
    def _clean(doc: dict) -> dict:
        """Remove Mongo's internal _id from output."""
        doc.pop("_id", None)
        return doc

    # ── audit events ─────────────────────────────────────────────────

    async def save_audit_event(self, event: dict) -> None:
        doc = {**event}
        doc["_id"] = doc["id"]
        await self.db.audit_events.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        await self._update_agent(event["agent_id"], event["decision"], event["action_name"])

    async def query_audit(self, *, agent_id=None, decision=None, action_name=None,
                          session_id=None, start_time=None, end_time=None,
                          limit=100, offset=0) -> list[dict]:
        filt = {}
        if agent_id:    filt["agent_id"] = agent_id
        if decision:    filt["decision"] = decision
        if action_name: filt["action_name"] = action_name
        if session_id:  filt["session_id"] = session_id
        if start_time or end_time:
            ts = {}
            if start_time: ts["$gte"] = start_time
            if end_time:   ts["$lte"] = end_time
            filt["timestamp"] = ts
        cursor = self.db.audit_events.find(filt).sort("timestamp", -1).skip(offset).limit(limit)
        return [self._clean(d) async for d in cursor]

    async def get_session_replay(self, session_id: str) -> list[dict]:
        cursor = self.db.audit_events.find({"session_id": session_id}).sort("timestamp", 1)
        return [self._clean(d) async for d in cursor]

    async def verify_chain(self) -> dict:
        cursor = self.db.audit_events.find({}, {"event_hash": 1, "previous_hash": 1}).sort("timestamp", 1)
        docs = [d async for d in cursor]
        if not docs:
            return {"valid": True, "count": 0, "break_index": None}
        prev = "GENESIS"
        for i, d in enumerate(docs):
            if d["previous_hash"] != prev:
                return {"valid": False, "count": len(docs), "break_index": i}
            prev = d["event_hash"]
        return {"valid": True, "count": len(docs), "break_index": None}

    # ── drift ────────────────────────────────────────────────────────

    async def save_drift_alert(self, alert: dict) -> None:
        doc = {**alert, "_id": alert["id"]}
        await self.db.drift_alerts.insert_one(doc)

    async def query_drift_alerts(self, *, agent_id=None, level=None, limit=50) -> list[dict]:
        filt = {}
        if agent_id: filt["agent_id"] = agent_id
        if level:    filt["alert_level"] = level
        cursor = self.db.drift_alerts.find(filt).sort("timestamp", -1).limit(limit)
        return [self._clean(d) async for d in cursor]

    # ── escalation ───────────────────────────────────────────────────

    async def save_escalation(self, escalation: dict) -> None:
        doc = {**escalation, "_id": escalation["id"]}
        doc.setdefault("created_at", datetime.utcnow().isoformat())
        doc.setdefault("status", "PENDING")
        await self.db.escalation_queue.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    async def get_escalations(self, status=None) -> list[dict]:
        filt = {"status": status} if status else {}
        cursor = self.db.escalation_queue.find(filt).sort("created_at", -1)
        return [self._clean(d) async for d in cursor]

    async def resolve_escalation(self, escalation_id: str, approved: bool, resolved_by="admin") -> Optional[dict]:
        status = "APPROVED" if approved else "REJECTED"
        now = datetime.utcnow().isoformat()
        await self.db.escalation_queue.update_one(
            {"_id": escalation_id},
            {"$set": {"status": status, "resolved_at": now, "resolved_by": resolved_by}},
        )
        doc = await self.db.escalation_queue.find_one({"_id": escalation_id})
        return self._clean(doc) if doc else None

    # ── agents ───────────────────────────────────────────────────────

    async def _update_agent(self, agent_id: str, decision: str, action_name: str) -> None:
        now = datetime.utcnow().isoformat()
        inc = {"total_actions": 1}
        if decision == "ALLOW":    inc["allowed"] = 1
        elif decision == "DENY":   inc["denied"] = 1
        elif decision == "ESCALATE": inc["escalated"] = 1

        await self.db.agent_profiles.update_one(
            {"_id": agent_id},
            {
                "$set": {"last_active": now},
                "$setOnInsert": {"first_seen": now, "agent_id": agent_id},
                "$inc": {**inc, f"action_distribution.{action_name}": 1},
            },
            upsert=True,
        )

    async def get_agents(self) -> list[dict]:
        cursor = self.db.agent_profiles.find().sort("last_active", -1)
        return [self._clean(d) async for d in cursor]

    # ── stats ────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        total     = await self.db.audit_events.count_documents({})
        allowed   = await self.db.audit_events.count_documents({"decision": "ALLOW"})
        denied    = await self.db.audit_events.count_documents({"decision": "DENY"})
        escalated = await self.db.audit_events.count_documents({"decision": "ESCALATE"})
        drifts    = await self.db.drift_alerts.count_documents({})
        agents    = await self.db.agent_profiles.count_documents({})
        chain     = await self.verify_chain()
        return {
            "total_actions": total, "allowed": allowed, "denied": denied,
            "escalated": escalated, "drift_alerts": drifts,
            "active_agents": agents, "chain_healthy": chain["valid"],
        }
