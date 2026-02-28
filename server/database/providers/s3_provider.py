"""
AgentVault — S3 Provider
Append-only audit log stored as JSON-lines in S3.

Uses an in-memory working set + periodic S3 flushes for write performance.
Supports: AWS S3, MinIO, DigitalOcean Spaces, any S3-compatible storage.

URI formats:
    s3://bucket-name/prefix
    s3://bucket-name/agentvault/audit
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import Any, Optional

from ..base import DatabaseProvider

logger = logging.getLogger("agentvault.db.s3")


class S3Provider(DatabaseProvider):
    """
    S3-based append-only audit store.

    Data layout in S3::

        s3://bucket/prefix/
            audit_events/YYYY-MM-DD/events.jsonl
            drift_alerts/YYYY-MM-DD/alerts.jsonl
            escalation_queue/escalations.json
            agent_profiles/profiles.json

    Install: ``pip install aioboto3``
    """

    def __init__(self, uri: str) -> None:
        # Parse s3://bucket/prefix
        clean = uri.replace("s3://", "")
        parts = clean.split("/", 1)
        self._bucket = parts[0]
        self._prefix = parts[1].rstrip("/") + "/" if len(parts) > 1 and parts[1] else ""
        self._session = None

        # In-memory working set (flushed to S3 periodically)
        self._audit_cache: list[dict] = []
        self._drift_cache: list[dict] = []
        self._escalations: list[dict] = []
        self._agents: dict[str, dict] = {}

    async def connect(self) -> None:
        try:
            import aioboto3
        except ImportError:
            raise ImportError(
                "aioboto3 is required for S3 support. "
                "Install it: pip install aioboto3"
            )
        self._session = aioboto3.Session()
        # Load existing state from S3 (if any)
        await self._load_state()
        logger.info("S3 provider connected — bucket=%s prefix=%s", self._bucket, self._prefix)

    async def close(self) -> None:
        await self._flush_all()

    async def health_check(self) -> dict[str, Any]:
        t0 = time.monotonic()
        async with self._session.client("s3") as s3:
            await s3.head_bucket(Bucket=self._bucket)
        return {
            "ok": True, "backend": "s3",
            "latency_ms": round((time.monotonic() - t0) * 1000, 2),
            "bucket": self._bucket,
        }

    # ── S3 I/O helpers ───────────────────────────────────────────────

    def _key(self, *parts: str) -> str:
        return self._prefix + "/".join(parts)

    async def _put_json(self, key: str, data: Any) -> None:
        async with self._session.client("s3") as s3:
            body = json.dumps(data, default=str).encode()
            await s3.put_object(Bucket=self._bucket, Key=key, Body=body, ContentType="application/json")

    async def _get_json(self, key: str) -> Any:
        try:
            async with self._session.client("s3") as s3:
                resp = await s3.get_object(Bucket=self._bucket, Key=key)
                body = await resp["Body"].read()
                return json.loads(body)
        except Exception:
            return None

    async def _append_jsonl(self, key: str, records: list[dict]) -> None:
        """Append JSON-lines to an existing file or create a new one."""
        existing = ""
        try:
            async with self._session.client("s3") as s3:
                resp = await s3.get_object(Bucket=self._bucket, Key=key)
                existing = (await resp["Body"].read()).decode()
        except Exception:
            pass
        lines = existing + "".join(json.dumps(r, default=str) + "\n" for r in records)
        async with self._session.client("s3") as s3:
            await s3.put_object(Bucket=self._bucket, Key=key, Body=lines.encode(), ContentType="application/x-ndjson")

    async def _load_state(self) -> None:
        """Load persistent state from S3."""
        esc = await self._get_json(self._key("escalation_queue", "escalations.json"))
        if esc:
            self._escalations = esc
        prof = await self._get_json(self._key("agent_profiles", "profiles.json"))
        if prof and isinstance(prof, dict):
            self._agents = prof

    async def _flush_all(self) -> None:
        """Flush all in-memory data to S3."""
        if self._audit_cache:
            day = datetime.utcnow().strftime("%Y-%m-%d")
            await self._append_jsonl(self._key("audit_events", day, "events.jsonl"), self._audit_cache)
            self._audit_cache.clear()
        if self._drift_cache:
            day = datetime.utcnow().strftime("%Y-%m-%d")
            await self._append_jsonl(self._key("drift_alerts", day, "alerts.jsonl"), self._drift_cache)
            self._drift_cache.clear()
        await self._put_json(self._key("escalation_queue", "escalations.json"), self._escalations)
        await self._put_json(self._key("agent_profiles", "profiles.json"), self._agents)

    # ── audit events ─────────────────────────────────────────────────

    async def save_audit_event(self, event: dict) -> None:
        self._audit_cache.append(event)
        await self._update_agent(event["agent_id"], event["decision"], event["action_name"])
        # Flush every 50 events
        if len(self._audit_cache) >= 50:
            await self._flush_all()

    async def query_audit(self, *, agent_id=None, decision=None, action_name=None,
                          session_id=None, start_time=None, end_time=None,
                          limit=100, offset=0) -> list[dict]:
        # For S3, we scan the in-memory cache. For full history, load from S3.
        results = list(self._audit_cache)
        if agent_id:    results = [e for e in results if e.get("agent_id") == agent_id]
        if decision:    results = [e for e in results if e.get("decision") == decision]
        if action_name: results = [e for e in results if e.get("action_name") == action_name]
        if session_id:  results = [e for e in results if e.get("session_id") == session_id]
        results.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return results[offset:offset + limit]

    async def get_session_replay(self, session_id: str) -> list[dict]:
        results = [e for e in self._audit_cache if e.get("session_id") == session_id]
        results.sort(key=lambda e: e.get("timestamp", ""))
        return results

    async def verify_chain(self) -> dict:
        events = sorted(self._audit_cache, key=lambda e: e.get("timestamp", ""))
        if not events:
            return {"valid": True, "count": 0, "break_index": None}
        prev = "GENESIS"
        for i, e in enumerate(events):
            if e.get("previous_hash") != prev:
                return {"valid": False, "count": len(events), "break_index": i}
            prev = e["event_hash"]
        return {"valid": True, "count": len(events), "break_index": None}

    # ── drift ────────────────────────────────────────────────────────

    async def save_drift_alert(self, alert: dict) -> None:
        self._drift_cache.append(alert)

    async def query_drift_alerts(self, *, agent_id=None, level=None, limit=50) -> list[dict]:
        results = list(self._drift_cache)
        if agent_id: results = [a for a in results if a.get("agent_id") == agent_id]
        if level:    results = [a for a in results if a.get("alert_level") == level]
        results.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
        return results[:limit]

    # ── escalation ───────────────────────────────────────────────────

    async def save_escalation(self, escalation: dict) -> None:
        escalation.setdefault("created_at", datetime.utcnow().isoformat())
        escalation.setdefault("status", "PENDING")
        # Replace if exists
        self._escalations = [e for e in self._escalations if e["id"] != escalation["id"]]
        self._escalations.append(escalation)
        await self._flush_all()

    async def get_escalations(self, status=None) -> list[dict]:
        results = self._escalations
        if status:
            results = [e for e in results if e.get("status") == status]
        return sorted(results, key=lambda e: e.get("created_at", ""), reverse=True)

    async def resolve_escalation(self, escalation_id: str, approved: bool, resolved_by="admin") -> Optional[dict]:
        for esc in self._escalations:
            if esc["id"] == escalation_id:
                esc["status"] = "APPROVED" if approved else "REJECTED"
                esc["resolved_at"] = datetime.utcnow().isoformat()
                esc["resolved_by"] = resolved_by
                await self._flush_all()
                return esc
        return None

    # ── agents ───────────────────────────────────────────────────────

    async def _update_agent(self, agent_id: str, decision: str, action_name: str) -> None:
        now = datetime.utcnow().isoformat()
        if agent_id not in self._agents:
            self._agents[agent_id] = {
                "agent_id": agent_id, "first_seen": now, "last_active": now,
                "total_actions": 0, "allowed": 0, "denied": 0, "escalated": 0,
                "action_distribution": {},
            }
        p = self._agents[agent_id]
        p["last_active"] = now
        p["total_actions"] = p.get("total_actions", 0) + 1
        if decision == "ALLOW":    p["allowed"] = p.get("allowed", 0) + 1
        elif decision == "DENY":   p["denied"] = p.get("denied", 0) + 1
        elif decision == "ESCALATE": p["escalated"] = p.get("escalated", 0) + 1
        dist = p.get("action_distribution", {})
        dist[action_name] = dist.get(action_name, 0) + 1
        p["action_distribution"] = dist

    async def get_agents(self) -> list[dict]:
        return sorted(self._agents.values(), key=lambda a: a.get("last_active", ""), reverse=True)

    # ── stats ────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        events = self._audit_cache
        chain = await self.verify_chain()
        return {
            "total_actions": len(events),
            "allowed": sum(1 for e in events if e.get("decision") == "ALLOW"),
            "denied": sum(1 for e in events if e.get("decision") == "DENY"),
            "escalated": sum(1 for e in events if e.get("decision") == "ESCALATE"),
            "drift_alerts": len(self._drift_cache),
            "active_agents": len(self._agents),
            "chain_healthy": chain["valid"],
        }
