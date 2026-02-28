"""
Shield Command — Cross-Agent Knowledge Distillation

When one agent learns something, the KnowledgeBroker propagates that
knowledge to all other agents that could benefit.

Example:
    Agent #3 fails on Stripe → learns "always include idempotency key"
    → KnowledgeBroker finds all agents using Stripe
    → injects the rule into their SemanticMemory
    → Agent #7 never makes that mistake

Integration points:
    - CognitiveMemory.SemanticMemory: reads/writes rules
    - ConnectorForge: identifies which agents use which connectors
    - MCPGateway: identifies which agents use which tools
"""

import json
import time
import uuid
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger("shield.knowledge_broker")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class AgentProfile:
    """Profile of a registered agent for knowledge routing."""
    agent_id: str
    name: str = ""
    tools: list[str] = field(default_factory=list)
    connectors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    registered_at: float = field(default_factory=time.time)


@dataclass
class PropagationRecord:
    """Record of a knowledge propagation event."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    source_agent: str = ""
    target_agent: str = ""
    rule_id: str = ""
    rule_text: str = ""
    confidence: float = 0.0
    propagated_at: float = field(default_factory=time.time)
    acknowledged: bool = False


# ---------------------------------------------------------------------------
# Knowledge Broker
# ---------------------------------------------------------------------------

class KnowledgeBroker:
    """
    Propagates learned knowledge across agents that share tools or connectors.
    
    Architecture:
        1. Maintains a registry of agent profiles (what tools/connectors they use)
        2. When new SemanticRules are created, identifies related agents
        3. Injects rules into target agents' SemanticMemory with provenance
        4. Tracks all propagation for audit and rollback
    
    Usage:
        broker = KnowledgeBroker()
        broker.register_agent(AgentProfile(agent_id="a1", tools=["github", "slack"]))
        broker.register_agent(AgentProfile(agent_id="a2", tools=["github", "jira"]))
        
        # When agent a1 learns something about GitHub:
        await broker.propagate(
            source_agent="a1",
            rule=SemanticRule(rule="Always check branch protection", related_tools=["github"]),
            memory_managers={"a2": a2_memory}
        )
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS agent_profiles (
        agent_id TEXT PRIMARY KEY,
        name TEXT DEFAULT '',
        tools TEXT DEFAULT '[]',
        connectors TEXT DEFAULT '[]',
        tags TEXT DEFAULT '[]',
        registered_at REAL
    );
    CREATE TABLE IF NOT EXISTS propagation_log (
        id TEXT PRIMARY KEY,
        source_agent TEXT,
        target_agent TEXT,
        rule_id TEXT,
        rule_text TEXT,
        confidence REAL,
        propagated_at REAL,
        acknowledged INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_prop_source ON propagation_log(source_agent);
    CREATE INDEX IF NOT EXISTS idx_prop_target ON propagation_log(target_agent);
    """

    def __init__(self, db_path: str = "shield_memory.db", decay_factor: float = 0.8):
        self.db_path = db_path
        self.decay_factor = decay_factor  # Confidence reduction when propagating
        self._agents: dict[str, AgentProfile] = {}
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── Agent Registration ────────────────────────────────────────

    def register_agent(self, profile: AgentProfile) -> None:
        """Register an agent profile for knowledge routing."""
        self._agents[profile.agent_id] = profile
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO agent_profiles 
                   (agent_id, name, tools, connectors, tags, registered_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (profile.agent_id, profile.name, json.dumps(profile.tools),
                 json.dumps(profile.connectors), json.dumps(profile.tags),
                 profile.registered_at),
            )
        logger.info(f"Registered agent '{profile.agent_id}' (tools={profile.tools})")

    def unregister_agent(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)
        with self._conn() as conn:
            conn.execute("DELETE FROM agent_profiles WHERE agent_id = ?", (agent_id,))

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        if agent_id in self._agents:
            return self._agents[agent_id]
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM agent_profiles WHERE agent_id = ?", (agent_id,)).fetchone()
        if row:
            profile = AgentProfile(
                agent_id=row[0], name=row[1], tools=json.loads(row[2] or "[]"),
                connectors=json.loads(row[3] or "[]"), tags=json.loads(row[4] or "[]"),
                registered_at=row[5],
            )
            self._agents[agent_id] = profile
            return profile
        return None

    def list_agents(self) -> list[AgentProfile]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM agent_profiles").fetchall()
        return [
            AgentProfile(
                agent_id=r[0], name=r[1], tools=json.loads(r[2] or "[]"),
                connectors=json.loads(r[3] or "[]"), tags=json.loads(r[4] or "[]"),
                registered_at=r[5],
            )
            for r in rows
        ]

    # ── Knowledge Propagation ─────────────────────────────────────

    def find_related_agents(
        self,
        source_agent: str,
        related_tools: list[str],
        related_connectors: list[str],
    ) -> list[AgentProfile]:
        """Find all agents that share tools or connectors with the source."""
        all_agents = self.list_agents()
        related = []
        
        for agent in all_agents:
            if agent.agent_id == source_agent:
                continue
            
            tool_overlap = set(agent.tools) & set(related_tools)
            connector_overlap = set(agent.connectors) & set(related_connectors)
            
            if tool_overlap or connector_overlap:
                related.append(agent)

        return related

    async def propagate(
        self,
        source_agent: str,
        rule,  # SemanticRule from cognitive_memory
        memory_managers: Optional[dict] = None,  # agent_id → CognitiveMemoryManager
    ) -> list[PropagationRecord]:
        """
        Propagate a learned rule to all related agents.
        
        Args:
            source_agent: ID of the agent that learned the rule
            rule: SemanticRule to propagate
            memory_managers: dict mapping agent_id to their CognitiveMemoryManager
        
        Returns:
            List of propagation records
        """
        related = self.find_related_agents(
            source_agent,
            related_tools=getattr(rule, "related_tools", []),
            related_connectors=getattr(rule, "related_connectors", []),
        )

        if not related:
            logger.debug(f"No related agents found for rule from {source_agent}")
            return []

        records = []
        for target in related:
            # Create propagated copy with reduced confidence
            propagated_confidence = getattr(rule, "confidence", 0.8) * self.decay_factor

            record = PropagationRecord(
                source_agent=source_agent,
                target_agent=target.agent_id,
                rule_id=getattr(rule, "id", ""),
                rule_text=getattr(rule, "rule", ""),
                confidence=propagated_confidence,
            )

            # If we have access to the target's memory, inject directly
            if memory_managers and target.agent_id in memory_managers:
                try:
                    from .cognitive_memory import SemanticRule as SR
                    propagated_rule = SR(
                        rule=getattr(rule, "rule", ""),
                        confidence=propagated_confidence,
                        evidence_count=1,
                        related_tools=getattr(rule, "related_tools", []),
                        related_connectors=getattr(rule, "related_connectors", []),
                        source=f"propagated:{source_agent}",
                    )
                    await memory_managers[target.agent_id].semantic.store(propagated_rule)
                    record.acknowledged = True
                except Exception as e:
                    logger.warning(f"Failed to inject rule into {target.agent_id}: {e}")

            # Log the propagation
            self._log_propagation(record)
            records.append(record)

        logger.info(
            f"Propagated rule from {source_agent} to {len(records)} agents: "
            f"{[r.target_agent for r in records]}"
        )
        return records

    # ── Propagation History ───────────────────────────────────────

    def _log_propagation(self, record: PropagationRecord):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO propagation_log 
                   (id, source_agent, target_agent, rule_id, rule_text, confidence, propagated_at, acknowledged)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (record.id, record.source_agent, record.target_agent,
                 record.rule_id, record.rule_text, record.confidence,
                 record.propagated_at, int(record.acknowledged)),
            )

    async def get_propagation_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[PropagationRecord]:
        with self._conn() as conn:
            if agent_id:
                rows = conn.execute(
                    """SELECT * FROM propagation_log 
                       WHERE source_agent = ? OR target_agent = ?
                       ORDER BY propagated_at DESC LIMIT ?""",
                    (agent_id, agent_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM propagation_log ORDER BY propagated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        
        return [
            PropagationRecord(
                id=r[0], source_agent=r[1], target_agent=r[2],
                rule_id=r[3], rule_text=r[4], confidence=r[5],
                propagated_at=r[6], acknowledged=bool(r[7]),
            )
            for r in rows
        ]

    async def get_stats(self) -> dict:
        with self._conn() as conn:
            total_agents = conn.execute("SELECT COUNT(*) FROM agent_profiles").fetchone()[0]
            total_propagations = conn.execute("SELECT COUNT(*) FROM propagation_log").fetchone()[0]
            acknowledged = conn.execute("SELECT COUNT(*) FROM propagation_log WHERE acknowledged = 1").fetchone()[0]
        
        return {
            "registered_agents": total_agents,
            "total_propagations": total_propagations,
            "acknowledged": acknowledged,
            "acknowledgment_rate": acknowledged / max(total_propagations, 1),
        }
