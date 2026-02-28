"""
AgentVault — Oracle Provider
Enterprise SQL backend via python-oracledb (async mode).

Supports: Oracle Cloud, AWS RDS for Oracle, on-premise Oracle DB.

URI formats:
    oracle://user:pass@host:1521/service_name
    oracle+tcp://user:pass@host:1521/service_name
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

from ..base import DatabaseProvider

logger = logging.getLogger("agentvault.db.oracle")

# Oracle uses different DDL syntax
_SCHEMA_CHECKS = [
    # (table_name, create_statement)
    ("audit_events", """
        CREATE TABLE audit_events (
            id VARCHAR2(128) PRIMARY KEY,
            timestamp_col VARCHAR2(64) NOT NULL,
            agent_id VARCHAR2(256) NOT NULL,
            session_id VARCHAR2(256) NOT NULL,
            trace_id VARCHAR2(256) NOT NULL,
            action_name VARCHAR2(512) NOT NULL,
            tool_name VARCHAR2(512) NOT NULL,
            parameters CLOB DEFAULT '{}',
            decision VARCHAR2(32) NOT NULL,
            reasoning CLOB DEFAULT '',
            confidence_score NUMBER,
            drift_score NUMBER,
            input_hash VARCHAR2(256) DEFAULT '',
            output_hash VARCHAR2(256) DEFAULT '',
            result CLOB,
            log_level VARCHAR2(32) DEFAULT 'standard',
            event_hash VARCHAR2(256) NOT NULL,
            previous_hash VARCHAR2(256) NOT NULL
        )
    """),
    ("drift_alerts", """
        CREATE TABLE drift_alerts (
            id VARCHAR2(128) PRIMARY KEY,
            agent_id VARCHAR2(256) NOT NULL,
            timestamp_col VARCHAR2(64) NOT NULL,
            deviation_score NUMBER NOT NULL,
            alert_level VARCHAR2(32) NOT NULL,
            baseline_distribution CLOB DEFAULT '{}',
            current_distribution CLOB DEFAULT '{}',
            window_size VARCHAR2(32) DEFAULT '1h',
            message CLOB DEFAULT ''
        )
    """),
    ("escalation_queue", """
        CREATE TABLE escalation_queue (
            id VARCHAR2(128) PRIMARY KEY,
            agent_id VARCHAR2(256) NOT NULL,
            session_id VARCHAR2(256) NOT NULL,
            trace_id VARCHAR2(256) NOT NULL,
            action_name VARCHAR2(512) NOT NULL,
            tool_name VARCHAR2(512) NOT NULL,
            parameters CLOB DEFAULT '{}',
            reasoning CLOB DEFAULT '',
            status VARCHAR2(32) DEFAULT 'PENDING',
            created_at VARCHAR2(64) NOT NULL,
            resolved_at VARCHAR2(64),
            resolved_by VARCHAR2(256)
        )
    """),
    ("agent_profiles", """
        CREATE TABLE agent_profiles (
            agent_id VARCHAR2(256) PRIMARY KEY,
            first_seen VARCHAR2(64) NOT NULL,
            last_active VARCHAR2(64) NOT NULL,
            total_actions NUMBER DEFAULT 0,
            allowed NUMBER DEFAULT 0,
            denied NUMBER DEFAULT 0,
            escalated NUMBER DEFAULT 0,
            action_distribution CLOB DEFAULT '{}'
        )
    """),
]


class OracleProvider(DatabaseProvider):
    """
    Oracle DB backend using python-oracledb (thin mode — no Oracle Client needed).

    Install: ``pip install oracledb``
    """

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._pool = None
        self._parse_uri()

    def _parse_uri(self) -> None:
        clean = self._uri.replace("oracle+tcp://", "oracle://")
        parsed = urlparse(clean)
        self._user = parsed.username or ""
        self._password = parsed.password or ""
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 1521
        self._service = parsed.path.lstrip("/") or "ORCL"

    async def connect(self) -> None:
        try:
            import oracledb
        except ImportError:
            raise ImportError(
                "oracledb is required for Oracle support. "
                "Install it: pip install oracledb"
            )
        oracledb.init_oracle_client()  # thin mode if client not available
        dsn = f"{self._host}:{self._port}/{self._service}"
        self._pool = oracledb.create_pool_async(
            user=self._user, password=self._password,
            dsn=dsn, min=2, max=10, increment=1,
        )
        await self._pool.open()

        # Create tables if not exist
        async with self._pool.acquire() as conn:
            for table_name, ddl in _SCHEMA_CHECKS:
                try:
                    cur = await conn.execute(
                        "SELECT COUNT(*) FROM user_tables WHERE table_name = :1",
                        [table_name.upper()],
                    )
                    row = await cur.fetchone()
                    if row[0] == 0:
                        await conn.execute(ddl)
                        await conn.commit()
                        logger.info("Created table: %s", table_name)
                except Exception as e:
                    logger.warning("Table check/create for %s: %s", table_name, e)

        logger.info("Oracle connected to %s:%s/%s", self._host, self._port, self._service)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def health_check(self) -> dict[str, Any]:
        t0 = time.monotonic()
        async with self._pool.acquire() as conn:
            cur = await conn.execute("SELECT 1 FROM DUAL")
            await cur.fetchone()
        return {
            "ok": True, "backend": "oracle",
            "latency_ms": round((time.monotonic() - t0) * 1000, 2),
            "service": self._service,
        }

    @property
    def pool(self):
        if not self._pool:
            raise RuntimeError("Not connected")
        return self._pool

    # ── audit events ─────────────────────────────────────────────────

    async def save_audit_event(self, event: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """MERGE INTO audit_events t USING (SELECT :1 AS id FROM DUAL) s ON (t.id = s.id)
                WHEN NOT MATCHED THEN INSERT
                (id, timestamp_col, agent_id, session_id, trace_id, action_name, tool_name,
                 parameters, decision, reasoning, confidence_score, drift_score,
                 input_hash, output_hash, result, log_level, event_hash, previous_hash)
                VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12,:13,:14,:15,:16,:17,:18)
                WHEN MATCHED THEN UPDATE SET decision=:9, reasoning=:10""",
                [
                    event["id"], event["timestamp"], event["agent_id"],
                    event["session_id"], event["trace_id"], event["action_name"],
                    event["tool_name"], json.dumps(event.get("parameters", {})),
                    event["decision"], event.get("reasoning", ""),
                    event.get("confidence_score"), event.get("drift_score"),
                    event.get("input_hash", ""), event.get("output_hash", ""),
                    json.dumps(event.get("result")) if event.get("result") else None,
                    event.get("log_level", "standard"),
                    event["event_hash"], event["previous_hash"],
                ],
            )
            await conn.commit()
            await self._update_agent(conn, event["agent_id"], event["decision"], event["action_name"])

    async def query_audit(self, *, agent_id=None, decision=None, action_name=None,
                          session_id=None, start_time=None, end_time=None,
                          limit=100, offset=0) -> list[dict]:
        conds, params = [], {}
        if agent_id:    conds.append("agent_id = :agent_id"); params["agent_id"] = agent_id
        if decision:    conds.append("decision = :decision"); params["decision"] = decision
        if action_name: conds.append("action_name = :action_name"); params["action_name"] = action_name
        if session_id:  conds.append("session_id = :session_id"); params["session_id"] = session_id
        if start_time:  conds.append("timestamp_col >= :start_time"); params["start_time"] = start_time
        if end_time:    conds.append("timestamp_col <= :end_time"); params["end_time"] = end_time
        where = " AND ".join(conds) or "1=1"
        params["lim"] = limit
        params["off"] = offset
        q = f"SELECT * FROM (SELECT a.*, ROWNUM rn FROM (SELECT * FROM audit_events WHERE {where} ORDER BY timestamp_col DESC) a WHERE ROWNUM <= :lim + :off) WHERE rn > :off"
        async with self.pool.acquire() as conn:
            cur = await conn.execute(q, params)
            cols = [c[0].lower() for c in cur.description]
            rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    async def get_session_replay(self, session_id: str) -> list[dict]:
        async with self.pool.acquire() as conn:
            cur = await conn.execute(
                "SELECT * FROM audit_events WHERE session_id = :1 ORDER BY timestamp_col ASC",
                [session_id],
            )
            cols = [c[0].lower() for c in cur.description]
            rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    async def verify_chain(self) -> dict:
        async with self.pool.acquire() as conn:
            cur = await conn.execute(
                "SELECT event_hash, previous_hash FROM audit_events ORDER BY timestamp_col ASC"
            )
            rows = await cur.fetchall()
        if not rows:
            return {"valid": True, "count": 0, "break_index": None}
        prev = "GENESIS"
        for i, (eh, ph) in enumerate(rows):
            if ph != prev:
                return {"valid": False, "count": len(rows), "break_index": i}
            prev = eh
        return {"valid": True, "count": len(rows), "break_index": None}

    # ── drift ────────────────────────────────────────────────────────

    async def save_drift_alert(self, alert: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO drift_alerts
                (id, agent_id, timestamp_col, deviation_score, alert_level,
                 baseline_distribution, current_distribution, window_size, message)
                VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9)""",
                [
                    alert["id"], alert["agent_id"], alert["timestamp"],
                    alert["deviation_score"], alert["alert_level"],
                    json.dumps(alert.get("baseline_distribution", {})),
                    json.dumps(alert.get("current_distribution", {})),
                    alert.get("window", "1h"), alert.get("message", ""),
                ],
            )
            await conn.commit()

    async def query_drift_alerts(self, *, agent_id=None, level=None, limit=50) -> list[dict]:
        conds, params = [], {}
        if agent_id: conds.append("agent_id = :agent_id"); params["agent_id"] = agent_id
        if level:    conds.append("alert_level = :level"); params["level"] = level
        where = " AND ".join(conds) or "1=1"
        params["lim"] = limit
        q = f"SELECT * FROM (SELECT * FROM drift_alerts WHERE {where} ORDER BY timestamp_col DESC) WHERE ROWNUM <= :lim"
        async with self.pool.acquire() as conn:
            cur = await conn.execute(q, params)
            cols = [c[0].lower() for c in cur.description]
            rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    # ── escalation ───────────────────────────────────────────────────

    async def save_escalation(self, escalation: dict) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """MERGE INTO escalation_queue t USING (SELECT :1 AS id FROM DUAL) s ON (t.id = s.id)
                WHEN NOT MATCHED THEN INSERT
                (id, agent_id, session_id, trace_id, action_name, tool_name,
                 parameters, reasoning, status, created_at, resolved_at, resolved_by)
                VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11,:12)
                WHEN MATCHED THEN UPDATE SET status=:9""",
                [
                    escalation["id"], escalation["agent_id"], escalation["session_id"],
                    escalation["trace_id"], escalation["action_name"], escalation["tool_name"],
                    json.dumps(escalation.get("parameters", {})),
                    escalation.get("reasoning", ""),
                    escalation.get("status", "PENDING"),
                    escalation.get("created_at", datetime.utcnow().isoformat()),
                    escalation.get("resolved_at"), escalation.get("resolved_by"),
                ],
            )
            await conn.commit()

    async def get_escalations(self, status=None) -> list[dict]:
        async with self.pool.acquire() as conn:
            if status:
                cur = await conn.execute(
                    "SELECT * FROM escalation_queue WHERE status = :1 ORDER BY created_at DESC",
                    [status],
                )
            else:
                cur = await conn.execute("SELECT * FROM escalation_queue ORDER BY created_at DESC")
            cols = [c[0].lower() for c in cur.description]
            rows = await cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]

    async def resolve_escalation(self, escalation_id: str, approved: bool, resolved_by="admin") -> Optional[dict]:
        status = "APPROVED" if approved else "REJECTED"
        now = datetime.utcnow().isoformat()
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE escalation_queue SET status=:1, resolved_at=:2, resolved_by=:3 WHERE id=:4",
                [status, now, resolved_by, escalation_id],
            )
            await conn.commit()
            cur = await conn.execute("SELECT * FROM escalation_queue WHERE id=:1", [escalation_id])
            cols = [c[0].lower() for c in cur.description]
            row = await cur.fetchone()
        return dict(zip(cols, row)) if row else None

    # ── agents ───────────────────────────────────────────────────────

    async def _update_agent(self, conn, agent_id: str, decision: str, action_name: str) -> None:
        now = datetime.utcnow().isoformat()
        cur = await conn.execute("SELECT * FROM agent_profiles WHERE agent_id = :1", [agent_id])
        cols = [c[0].lower() for c in cur.description] if cur.description else []
        row = await cur.fetchone()
        if row:
            p = dict(zip(cols, row))
            dist = json.loads(p.get("action_distribution", "{}"))
            dist[action_name] = dist.get(action_name, 0) + 1
            await conn.execute(
                """UPDATE agent_profiles SET last_active=:1, total_actions=total_actions+1,
                allowed=allowed+:2, denied=denied+:3, escalated=escalated+:4,
                action_distribution=:5 WHERE agent_id=:6""",
                [now, int(decision=="ALLOW"), int(decision=="DENY"), int(decision=="ESCALATE"),
                 json.dumps(dist), agent_id],
            )
        else:
            await conn.execute(
                """INSERT INTO agent_profiles
                (agent_id, first_seen, last_active, total_actions, allowed, denied, escalated, action_distribution)
                VALUES (:1,:2,:3,1,:4,:5,:6,:7)""",
                [agent_id, now, now, int(decision=="ALLOW"), int(decision=="DENY"),
                 int(decision=="ESCALATE"), json.dumps({action_name: 1})],
            )
        await conn.commit()

    async def get_agents(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            cur = await conn.execute("SELECT * FROM agent_profiles ORDER BY last_active DESC")
            cols = [c[0].lower() for c in cur.description]
            rows = await cur.fetchall()
        results = []
        for r in rows:
            d = dict(zip(cols, r))
            if isinstance(d.get("action_distribution"), str):
                d["action_distribution"] = json.loads(d["action_distribution"])
            results.append(d)
        return results

    # ── stats ────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        async with self.pool.acquire() as conn:
            total     = (await (await conn.execute("SELECT COUNT(*) FROM audit_events")).fetchone())[0]
            allowed   = (await (await conn.execute("SELECT COUNT(*) FROM audit_events WHERE decision='ALLOW'")).fetchone())[0]
            denied    = (await (await conn.execute("SELECT COUNT(*) FROM audit_events WHERE decision='DENY'")).fetchone())[0]
            escalated = (await (await conn.execute("SELECT COUNT(*) FROM audit_events WHERE decision='ESCALATE'")).fetchone())[0]
            drifts    = (await (await conn.execute("SELECT COUNT(*) FROM drift_alerts")).fetchone())[0]
            agents    = (await (await conn.execute("SELECT COUNT(*) FROM agent_profiles")).fetchone())[0]
        chain = await self.verify_chain()
        return {
            "total_actions": total, "allowed": allowed, "denied": denied,
            "escalated": escalated, "drift_alerts": drifts,
            "active_agents": agents, "chain_healthy": chain["valid"],
        }
