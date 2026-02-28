"""
AgentVault — SQLite Provider
Local async SQLite backend (default fallback). Migrated from the old database.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Optional

import aiosqlite

from ..base import DatabaseProvider

logger = logging.getLogger("agentvault.db.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    action_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    parameters TEXT DEFAULT '{}',
    decision TEXT NOT NULL,
    reasoning TEXT DEFAULT '',
    confidence_score REAL,
    drift_score REAL,
    input_hash TEXT DEFAULT '',
    output_hash TEXT DEFAULT '',
    result TEXT,
    log_level TEXT DEFAULT 'standard',
    event_hash TEXT NOT NULL,
    previous_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_decision ON audit_events(decision);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_events(action_name);

CREATE TABLE IF NOT EXISTS drift_alerts (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    deviation_score REAL NOT NULL,
    alert_level TEXT NOT NULL,
    baseline_distribution TEXT DEFAULT '{}',
    current_distribution TEXT DEFAULT '{}',
    window TEXT DEFAULT '1h',
    message TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_drift_agent ON drift_alerts(agent_id);
CREATE INDEX IF NOT EXISTS idx_drift_level ON drift_alerts(alert_level);

CREATE TABLE IF NOT EXISTS escalation_queue (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    action_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    parameters TEXT DEFAULT '{}',
    reasoning TEXT DEFAULT '',
    status TEXT DEFAULT 'PENDING',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolved_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_esc_status ON escalation_queue(status);

CREATE TABLE IF NOT EXISTS agent_profiles (
    agent_id TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,
    last_active TEXT NOT NULL,
    total_actions INTEGER DEFAULT 0,
    allowed INTEGER DEFAULT 0,
    denied INTEGER DEFAULT 0,
    escalated INTEGER DEFAULT 0,
    action_distribution TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT DEFAULT '',
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    name TEXT DEFAULT 'Default',
    prefix TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    last_used TEXT,
    revoked INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
"""


class SQLiteProvider(DatabaseProvider):
    """Async SQLite backend — zero external dependencies beyond aiosqlite."""

    def __init__(self, uri: str) -> None:
        # Handle both "sqlite:///path" and bare "path"
        self._path = uri.replace("sqlite:///", "").replace("sqlite://", "") or "agentvault.db"
        self._db: Optional[aiosqlite.Connection] = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info("SQLite connected at %s", self._path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            logger.info("SQLite connection closed")

    async def health_check(self) -> dict[str, Any]:
        t0 = time.monotonic()
        cursor = await self._db.execute("SELECT 1")
        await cursor.fetchone()
        return {"ok": True, "backend": "sqlite", "latency_ms": round((time.monotonic() - t0) * 1000, 2), "path": self._path}

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def db(self) -> aiosqlite.Connection:
        if not self._db:
            raise RuntimeError("Not connected")
        return self._db

    @staticmethod
    def _row(row) -> dict:
        return dict(row) if row else {}

    # ── audit events ─────────────────────────────────────────────────

    async def save_audit_event(self, event: dict) -> None:
        await self.db.execute(
            """INSERT OR REPLACE INTO audit_events
            (id, timestamp, agent_id, session_id, trace_id, action_name, tool_name,
             parameters, decision, reasoning, confidence_score, drift_score,
             input_hash, output_hash, result, log_level, event_hash, previous_hash)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event["id"], event["timestamp"], event["agent_id"],
                event["session_id"], event["trace_id"], event["action_name"],
                event["tool_name"], json.dumps(event.get("parameters", {})),
                event["decision"], event.get("reasoning", ""),
                event.get("confidence_score"), event.get("drift_score"),
                event.get("input_hash", ""), event.get("output_hash", ""),
                json.dumps(event.get("result")) if event.get("result") else None,
                event.get("log_level", "standard"),
                event["event_hash"], event["previous_hash"],
            ),
        )
        await self.db.commit()
        await self._update_agent(event["agent_id"], event["decision"], event["action_name"])

    async def query_audit(
        self, *, agent_id=None, decision=None, action_name=None,
        session_id=None, start_time=None, end_time=None,
        limit=100, offset=0,
    ) -> list[dict]:
        conds, params = [], []
        if agent_id:    conds.append("agent_id = ?");    params.append(agent_id)
        if decision:    conds.append("decision = ?");    params.append(decision)
        if action_name: conds.append("action_name = ?"); params.append(action_name)
        if session_id:  conds.append("session_id = ?");  params.append(session_id)
        if start_time:  conds.append("timestamp >= ?");  params.append(start_time)
        if end_time:    conds.append("timestamp <= ?");  params.append(end_time)
        where = " AND ".join(conds) or "1=1"
        params += [limit, offset]
        cur = await self.db.execute(
            f"SELECT * FROM audit_events WHERE {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?", params
        )
        return [self._row(r) for r in await cur.fetchall()]

    async def get_session_replay(self, session_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT * FROM audit_events WHERE session_id = ? ORDER BY timestamp ASC", (session_id,)
        )
        return [self._row(r) for r in await cur.fetchall()]

    async def verify_chain(self) -> dict:
        cur = await self.db.execute("SELECT * FROM audit_events ORDER BY timestamp ASC")
        rows = await cur.fetchall()
        if not rows:
            return {"valid": True, "count": 0, "break_index": None}
        prev = "GENESIS"
        for i, row in enumerate(rows):
            e = self._row(row)
            if e["previous_hash"] != prev:
                return {"valid": False, "count": len(rows), "break_index": i}
            prev = e["event_hash"]
        return {"valid": True, "count": len(rows), "break_index": None}

    # ── drift ────────────────────────────────────────────────────────

    async def save_drift_alert(self, alert: dict) -> None:
        await self.db.execute(
            """INSERT INTO drift_alerts
            (id, agent_id, timestamp, deviation_score, alert_level,
             baseline_distribution, current_distribution, window, message)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                alert["id"], alert["agent_id"], alert["timestamp"],
                alert["deviation_score"], alert["alert_level"],
                json.dumps(alert.get("baseline_distribution", {})),
                json.dumps(alert.get("current_distribution", {})),
                alert.get("window", "1h"), alert.get("message", ""),
            ),
        )
        await self.db.commit()

    async def query_drift_alerts(self, *, agent_id=None, level=None, limit=50) -> list[dict]:
        conds, params = [], []
        if agent_id: conds.append("agent_id = ?"); params.append(agent_id)
        if level:    conds.append("alert_level = ?"); params.append(level)
        where = " AND ".join(conds) or "1=1"
        params.append(limit)
        cur = await self.db.execute(
            f"SELECT * FROM drift_alerts WHERE {where} ORDER BY timestamp DESC LIMIT ?", params
        )
        return [self._row(r) for r in await cur.fetchall()]

    # ── escalation ───────────────────────────────────────────────────

    async def save_escalation(self, escalation: dict) -> None:
        await self.db.execute(
            """INSERT OR REPLACE INTO escalation_queue
            (id, agent_id, session_id, trace_id, action_name, tool_name,
             parameters, reasoning, status, created_at, resolved_at, resolved_by)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                escalation["id"], escalation["agent_id"], escalation["session_id"],
                escalation["trace_id"], escalation["action_name"], escalation["tool_name"],
                json.dumps(escalation.get("parameters", {})),
                escalation.get("reasoning", ""),
                escalation.get("status", "PENDING"),
                escalation.get("created_at", datetime.utcnow().isoformat()),
                escalation.get("resolved_at"), escalation.get("resolved_by"),
            ),
        )
        await self.db.commit()

    async def get_escalations(self, status=None) -> list[dict]:
        if status:
            cur = await self.db.execute(
                "SELECT * FROM escalation_queue WHERE status = ? ORDER BY created_at DESC", (status,)
            )
        else:
            cur = await self.db.execute("SELECT * FROM escalation_queue ORDER BY created_at DESC")
        return [self._row(r) for r in await cur.fetchall()]

    async def resolve_escalation(self, escalation_id: str, approved: bool, resolved_by="admin") -> Optional[dict]:
        status = "APPROVED" if approved else "REJECTED"
        now = datetime.utcnow().isoformat()
        await self.db.execute(
            "UPDATE escalation_queue SET status=?, resolved_at=?, resolved_by=? WHERE id=?",
            (status, now, resolved_by, escalation_id),
        )
        await self.db.commit()
        cur = await self.db.execute("SELECT * FROM escalation_queue WHERE id=?", (escalation_id,))
        row = await cur.fetchone()
        return self._row(row) if row else None

    # ── agents ───────────────────────────────────────────────────────

    async def _update_agent(self, agent_id: str, decision: str, action_name: str) -> None:
        now = datetime.utcnow().isoformat()
        cur = await self.db.execute("SELECT * FROM agent_profiles WHERE agent_id = ?", (agent_id,))
        row = await cur.fetchone()
        if row:
            p = self._row(row)
            dist = json.loads(p.get("action_distribution", "{}"))
            dist[action_name] = dist.get(action_name, 0) + 1
            inc = {"allowed": int(decision == "ALLOW"), "denied": int(decision == "DENY"), "escalated": int(decision == "ESCALATE")}
            await self.db.execute(
                """UPDATE agent_profiles SET last_active=?, total_actions=total_actions+1,
                allowed=allowed+?, denied=denied+?, escalated=escalated+?,
                action_distribution=? WHERE agent_id=?""",
                (now, inc["allowed"], inc["denied"], inc["escalated"], json.dumps(dist), agent_id),
            )
        else:
            dist = {action_name: 1}
            await self.db.execute(
                """INSERT INTO agent_profiles
                (agent_id, first_seen, last_active, total_actions, allowed, denied, escalated, action_distribution)
                VALUES (?,?,?,1,?,?,?,?)""",
                (agent_id, now, now, int(decision=="ALLOW"), int(decision=="DENY"), int(decision=="ESCALATE"), json.dumps(dist)),
            )
        await self.db.commit()

    async def get_agents(self) -> list[dict]:
        cur = await self.db.execute("SELECT * FROM agent_profiles ORDER BY last_active DESC")
        results = []
        for row in await cur.fetchall():
            d = self._row(row)
            d["action_distribution"] = json.loads(d.get("action_distribution", "{}"))
            results.append(d)
        return results

    # ── stats ────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        db = self.db
        total    = (await (await db.execute("SELECT COUNT(*) FROM audit_events")).fetchone())[0]
        allowed  = (await (await db.execute("SELECT COUNT(*) FROM audit_events WHERE decision='ALLOW'")).fetchone())[0]
        denied   = (await (await db.execute("SELECT COUNT(*) FROM audit_events WHERE decision='DENY'")).fetchone())[0]
        escalated= (await (await db.execute("SELECT COUNT(*) FROM audit_events WHERE decision='ESCALATE'")).fetchone())[0]
        drifts   = (await (await db.execute("SELECT COUNT(*) FROM drift_alerts")).fetchone())[0]
        agents   = (await (await db.execute("SELECT COUNT(*) FROM agent_profiles")).fetchone())[0]
        chain    = await self.verify_chain()
        return {
            "total_actions": total, "allowed": allowed, "denied": denied,
            "escalated": escalated, "drift_alerts": drifts,
            "active_agents": agents, "chain_healthy": chain["valid"],
        }

    # ── users ────────────────────────────────────────────────────────

    async def create_user(self, user: dict) -> None:
        await self.db.execute(
            "INSERT INTO users (id, email, name, password_hash, created_at) VALUES (?,?,?,?,?)",
            (user["id"], user["email"], user.get("name", ""), user["password_hash"], user["created_at"]),
        )
        await self.db.commit()

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        cur = await self.db.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = await cur.fetchone()
        return self._row(row) if row else None

    # ── API keys ─────────────────────────────────────────────────────

    async def save_api_key(self, key_data: dict) -> None:
        await self.db.execute(
            "INSERT INTO api_keys (id, user_id, key_hash, name, prefix, created_at) VALUES (?,?,?,?,?,?)",
            (key_data["id"], key_data["user_id"], key_data["key_hash"],
             key_data["name"], key_data["prefix"], key_data["created_at"]),
        )
        await self.db.commit()

    async def get_api_keys(self, user_id: str) -> list[dict]:
        cur = await self.db.execute(
            "SELECT id, name, prefix, created_at, last_used FROM api_keys WHERE user_id = ? AND revoked = 0 ORDER BY created_at DESC",
            (user_id,),
        )
        return [self._row(r) for r in await cur.fetchall()]

    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        cur = await self.db.execute(
            "UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_id = ?",
            (key_id, user_id),
        )
        await self.db.commit()
        return cur.rowcount > 0

    async def verify_api_key(self, raw_key: str) -> Optional[dict]:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        cur = await self.db.execute(
            """SELECT ak.user_id, u.email, u.name FROM api_keys ak
            JOIN users u ON u.id = ak.user_id
            WHERE ak.key_hash = ? AND ak.revoked = 0""",
            (key_hash,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        d = self._row(row)
        # Update last_used
        await self.db.execute(
            "UPDATE api_keys SET last_used = ? WHERE key_hash = ?",
            (datetime.utcnow().isoformat(), key_hash),
        )
        await self.db.commit()
        return {"id": d["user_id"], "email": d["email"], "name": d.get("name", "")}
