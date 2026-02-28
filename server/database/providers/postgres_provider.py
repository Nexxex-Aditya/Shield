"""
AgentVault — PostgreSQL Provider
Cloud-grade SQL backend with connection pooling via asyncpg.

Supports: AWS RDS, Azure Database, GCP Cloud SQL, Supabase, self-hosted.

URI formats:
    postgresql://user:pass@host:5432/dbname
    postgres://user:pass@host/dbname?sslmode=require
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Optional

from ..base import DatabaseProvider

logger = logging.getLogger("agentvault.db.postgres")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    action_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    parameters JSONB DEFAULT '{}',
    decision TEXT NOT NULL,
    reasoning TEXT DEFAULT '',
    confidence_score DOUBLE PRECISION,
    drift_score DOUBLE PRECISION,
    input_hash TEXT DEFAULT '',
    output_hash TEXT DEFAULT '',
    result JSONB,
    log_level TEXT DEFAULT 'standard',
    event_hash TEXT NOT NULL,
    previous_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pg_audit_agent ON audit_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_pg_audit_ts ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_pg_audit_decision ON audit_events(decision);
CREATE INDEX IF NOT EXISTS idx_pg_audit_action ON audit_events(action_name);
CREATE INDEX IF NOT EXISTS idx_pg_audit_session ON audit_events(session_id);

CREATE TABLE IF NOT EXISTS drift_alerts (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    deviation_score DOUBLE PRECISION NOT NULL,
    alert_level TEXT NOT NULL,
    baseline_distribution JSONB DEFAULT '{}',
    current_distribution JSONB DEFAULT '{}',
    window TEXT DEFAULT '1h',
    message TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_pg_drift_agent ON drift_alerts(agent_id);

CREATE TABLE IF NOT EXISTS escalation_queue (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    action_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    parameters JSONB DEFAULT '{}',
    reasoning TEXT DEFAULT '',
    status TEXT DEFAULT 'PENDING',
    created_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_pg_esc_status ON escalation_queue(status);

CREATE TABLE IF NOT EXISTS agent_profiles (
    agent_id TEXT PRIMARY KEY,
    first_seen TIMESTAMPTZ NOT NULL,
    last_active TIMESTAMPTZ NOT NULL,
    total_actions INTEGER DEFAULT 0,
    allowed INTEGER DEFAULT 0,
    denied INTEGER DEFAULT 0,
    escalated INTEGER DEFAULT 0,
    action_distribution JSONB DEFAULT '{}'
);
"""


class PostgresProvider(DatabaseProvider):
    """
    PostgreSQL backend using asyncpg connection pool.

    Install: ``pip install asyncpg``
    """

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._pool = None

    async def connect(self) -> None:
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgreSQL support. "
                "Install it: pip install asyncpg"
            )
        self._pool = await asyncpg.create_pool(
            self._uri, min_size=2, max_size=10,
            command_timeout=30,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA)
        logger.info("PostgreSQL pool created — %s", self._uri.split("@")[-1] if "@" in self._uri else "local")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def health_check(self) -> dict[str, Any]:
        t0 = time.monotonic()
        async with self._pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {
            "ok": True, "backend": "postgresql",
            "latency_ms": round((time.monotonic() - t0) * 1000, 2),
            "pool_size": self._pool.get_size(),
        }

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def pool(self):
        if not self._pool:
            raise RuntimeError("Not connected")
        return self._pool

    # ── audit events ─────────────────────────────────────────────────

    async def save_audit_event(self, event: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO audit_events
                (id, timestamp, agent_id, session_id, trace_id, action_name, tool_name,
                 parameters, decision, reasoning, confidence_score, drift_score,
                 input_hash, output_hash, result, log_level, event_hash, previous_hash)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
                ON CONFLICT (id) DO UPDATE SET decision=$9, reasoning=$10""",
                event["id"], event["timestamp"], event["agent_id"],
                event["session_id"], event["trace_id"], event["action_name"],
                event["tool_name"], json.dumps(event.get("parameters", {})),
                event["decision"], event.get("reasoning", ""),
                event.get("confidence_score"), event.get("drift_score"),
                event.get("input_hash", ""), event.get("output_hash", ""),
                json.dumps(event.get("result")) if event.get("result") else None,
                event.get("log_level", "standard"),
                event["event_hash"], event["previous_hash"],
            )
            await self._update_agent(conn, event["agent_id"], event["decision"], event["action_name"])

    async def query_audit(self, *, agent_id=None, decision=None, action_name=None,
                          session_id=None, start_time=None, end_time=None,
                          limit=100, offset=0) -> list[dict]:
        conds, params, idx = [], [], 0
        def _add(col, val):
            nonlocal idx; idx += 1; conds.append(f"{col} = ${idx}"); params.append(val)
        if agent_id:    _add("agent_id", agent_id)
        if decision:    _add("decision", decision)
        if action_name: _add("action_name", action_name)
        if session_id:  _add("session_id", session_id)
        if start_time:  idx += 1; conds.append(f"timestamp >= ${idx}"); params.append(start_time)
        if end_time:    idx += 1; conds.append(f"timestamp <= ${idx}"); params.append(end_time)
        where = " AND ".join(conds) or "TRUE"
        idx += 1; params.append(limit)
        idx += 1; params.append(offset)
        q = f"SELECT * FROM audit_events WHERE {where} ORDER BY timestamp DESC LIMIT ${idx-1} OFFSET ${idx}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, *params)
        return [dict(r) for r in rows]

    async def get_session_replay(self, session_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM audit_events WHERE session_id=$1 ORDER BY timestamp ASC", session_id
            )
        return [dict(r) for r in rows]

    async def verify_chain(self) -> dict:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT event_hash, previous_hash FROM audit_events ORDER BY timestamp ASC")
        if not rows:
            return {"valid": True, "count": 0, "break_index": None}
        prev = "GENESIS"
        for i, r in enumerate(rows):
            if r["previous_hash"] != prev:
                return {"valid": False, "count": len(rows), "break_index": i}
            prev = r["event_hash"]
        return {"valid": True, "count": len(rows), "break_index": None}

    # ── drift ────────────────────────────────────────────────────────

    async def save_drift_alert(self, alert: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO drift_alerts
                (id, agent_id, timestamp, deviation_score, alert_level,
                 baseline_distribution, current_distribution, window, message)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                alert["id"], alert["agent_id"], alert["timestamp"],
                alert["deviation_score"], alert["alert_level"],
                json.dumps(alert.get("baseline_distribution", {})),
                json.dumps(alert.get("current_distribution", {})),
                alert.get("window", "1h"), alert.get("message", ""),
            )

    async def query_drift_alerts(self, *, agent_id=None, level=None, limit=50) -> list[dict]:
        conds, params, idx = [], [], 0
        if agent_id: idx += 1; conds.append(f"agent_id=${idx}"); params.append(agent_id)
        if level:    idx += 1; conds.append(f"alert_level=${idx}"); params.append(level)
        where = " AND ".join(conds) or "TRUE"
        idx += 1; params.append(limit)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM drift_alerts WHERE {where} ORDER BY timestamp DESC LIMIT ${idx}", *params
            )
        return [dict(r) for r in rows]

    # ── escalation ───────────────────────────────────────────────────

    async def save_escalation(self, escalation: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO escalation_queue
                (id, agent_id, session_id, trace_id, action_name, tool_name,
                 parameters, reasoning, status, created_at, resolved_at, resolved_by)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (id) DO UPDATE SET status=$9""",
                escalation["id"], escalation["agent_id"], escalation["session_id"],
                escalation["trace_id"], escalation["action_name"], escalation["tool_name"],
                json.dumps(escalation.get("parameters", {})),
                escalation.get("reasoning", ""),
                escalation.get("status", "PENDING"),
                escalation.get("created_at", datetime.utcnow().isoformat()),
                escalation.get("resolved_at"), escalation.get("resolved_by"),
            )

    async def get_escalations(self, status=None) -> list[dict]:
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch(
                    "SELECT * FROM escalation_queue WHERE status=$1 ORDER BY created_at DESC", status
                )
            else:
                rows = await conn.fetch("SELECT * FROM escalation_queue ORDER BY created_at DESC")
        return [dict(r) for r in rows]

    async def resolve_escalation(self, escalation_id: str, approved: bool, resolved_by="admin") -> Optional[dict]:
        status = "APPROVED" if approved else "REJECTED"
        now = datetime.utcnow().isoformat()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE escalation_queue SET status=$1, resolved_at=$2, resolved_by=$3 WHERE id=$4",
                status, now, resolved_by, escalation_id,
            )
            row = await conn.fetchrow("SELECT * FROM escalation_queue WHERE id=$1", escalation_id)
        return dict(row) if row else None

    # ── agents ───────────────────────────────────────────────────────

    async def _update_agent(self, conn, agent_id: str, decision: str, action_name: str) -> None:
        now = datetime.utcnow().isoformat()
        row = await conn.fetchrow("SELECT * FROM agent_profiles WHERE agent_id=$1", agent_id)
        if row:
            dist = json.loads(row["action_distribution"]) if isinstance(row["action_distribution"], str) else dict(row["action_distribution"])
            dist[action_name] = dist.get(action_name, 0) + 1
            await conn.execute(
                """UPDATE agent_profiles SET last_active=$1, total_actions=total_actions+1,
                allowed=allowed+$2, denied=denied+$3, escalated=escalated+$4,
                action_distribution=$5 WHERE agent_id=$6""",
                now, int(decision=="ALLOW"), int(decision=="DENY"), int(decision=="ESCALATE"),
                json.dumps(dist), agent_id,
            )
        else:
            await conn.execute(
                """INSERT INTO agent_profiles
                (agent_id, first_seen, last_active, total_actions, allowed, denied, escalated, action_distribution)
                VALUES ($1,$2,$3,1,$4,$5,$6,$7)""",
                agent_id, now, now, int(decision=="ALLOW"), int(decision=="DENY"),
                int(decision=="ESCALATE"), json.dumps({action_name: 1}),
            )

    async def get_agents(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM agent_profiles ORDER BY last_active DESC")
        results = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("action_distribution"), str):
                d["action_distribution"] = json.loads(d["action_distribution"])
            results.append(d)
        return results

    # ── stats ────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        async with self.pool.acquire() as conn:
            total     = await conn.fetchval("SELECT COUNT(*) FROM audit_events")
            allowed   = await conn.fetchval("SELECT COUNT(*) FROM audit_events WHERE decision='ALLOW'")
            denied    = await conn.fetchval("SELECT COUNT(*) FROM audit_events WHERE decision='DENY'")
            escalated = await conn.fetchval("SELECT COUNT(*) FROM audit_events WHERE decision='ESCALATE'")
            drifts    = await conn.fetchval("SELECT COUNT(*) FROM drift_alerts")
            agents    = await conn.fetchval("SELECT COUNT(*) FROM agent_profiles")
        chain = await self.verify_chain()
        return {
            "total_actions": total, "allowed": allowed, "denied": denied,
            "escalated": escalated, "drift_alerts": drifts,
            "active_agents": agents, "chain_healthy": chain["valid"],
        }
