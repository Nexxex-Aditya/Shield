"""
Shield Command — Intent Mesh Network

Agents broadcast capabilities. When one agent needs help with a task
outside its expertise, it queries the mesh and finds another agent
that can handle the sub-task. They negotiate a contract enforced
by MCP Gateway.

Architecture:
    - AgentManifest: declares capabilities, tools, trust, current load
    - IntentMesh: semantic matching + contract negotiation
    - CollaborationContract: data sharing rules, permissions, accountability
    - Each collaboration is security-governed via MCPGateway

Result: agents self-organize into teams based on what needs to get done.
"""

import json
import time
import uuid
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable
from enum import Enum

logger = logging.getLogger("shield.intent_mesh")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class ContractStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class AgentManifest:
    """What an agent broadcasts to the mesh."""
    agent_id: str
    name: str = ""
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    connectors: list[str] = field(default_factory=list)
    trust_score: float = 0.8
    current_load: float = 0.0        # 0=idle, 1=fully loaded
    max_concurrent: int = 5
    active_tasks: int = 0
    specializations: list[str] = field(default_factory=list)  # e.g., ["code_review", "data_analysis"]
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskOffer:
    """An agent's offer to handle a task."""
    agent_id: str
    accepted: bool = False
    estimated_cost: float = 0.0
    estimated_time_ms: float = 0.0
    confidence: float = 0.0
    conditions: list[str] = field(default_factory=list)  # "needs read access to DB"
    reason: str = ""


@dataclass
class CollaborationContract:
    """Governs a collaboration between two agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    requester_id: str = ""
    provider_id: str = ""
    task: str = ""
    status: ContractStatus = ContractStatus.PROPOSED
    
    # Permissions
    data_sharing: list[str] = field(default_factory=list)  # what data can be shared
    allowed_tools: list[str] = field(default_factory=list)  # what tools the provider can use
    forbidden_tools: list[str] = field(default_factory=list)
    max_cost: float = 0.0
    max_time_ms: float = 30000.0
    
    # Results
    result: Any = None
    error: Optional[str] = None
    actual_cost: float = 0.0
    actual_time_ms: float = 0.0
    
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class AgentMatch:
    """Result of a mesh search."""
    provider: AgentManifest
    offer: TaskOffer
    contract: CollaborationContract
    match_score: float = 0.0


# ---------------------------------------------------------------------------
# Intent Mesh Network
# ---------------------------------------------------------------------------

class IntentMesh:
    """
    Self-organizing multi-agent discovery and collaboration network.
    
    Agents register their capabilities. When an agent needs help, it
    queries the mesh. The mesh finds the best match, negotiates a
    contract, and facilitates the collaboration.
    
    Usage:
        mesh = IntentMesh()
        mesh.register(AgentManifest(
            agent_id="agent-code",
            capabilities=["code_review", "refactoring", "testing"],
            tools=["github"],
        ))
        mesh.register(AgentManifest(
            agent_id="agent-data",
            capabilities=["sql_queries", "data_analysis", "visualization"],
            tools=["postgresql"],
        ))
        
        # Agent-code needs data analysis
        match = await mesh.request_help(
            requester="agent-code",
            task="Analyze the database performance metrics for the last month",
        )
        # match.provider = agent-data
        # match.contract = governed collaboration agreement
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS mesh_manifests (
        agent_id TEXT PRIMARY KEY,
        name TEXT,
        capabilities TEXT DEFAULT '[]',
        tools TEXT DEFAULT '[]',
        connectors TEXT DEFAULT '[]',
        trust_score REAL DEFAULT 0.8,
        current_load REAL DEFAULT 0.0,
        max_concurrent INTEGER DEFAULT 5,
        active_tasks INTEGER DEFAULT 0,
        specializations TEXT DEFAULT '[]',
        registered_at REAL,
        last_heartbeat REAL,
        metadata TEXT DEFAULT '{}'
    );
    CREATE TABLE IF NOT EXISTS mesh_contracts (
        id TEXT PRIMARY KEY,
        requester_id TEXT,
        provider_id TEXT,
        task TEXT,
        status TEXT DEFAULT 'proposed',
        data_sharing TEXT DEFAULT '[]',
        allowed_tools TEXT DEFAULT '[]',
        forbidden_tools TEXT DEFAULT '[]',
        max_cost REAL DEFAULT 0.0,
        max_time_ms REAL DEFAULT 30000,
        result TEXT,
        error TEXT,
        actual_cost REAL DEFAULT 0.0,
        actual_time_ms REAL DEFAULT 0.0,
        created_at REAL,
        completed_at REAL
    );
    CREATE INDEX IF NOT EXISTS idx_mesh_status ON mesh_contracts(status);
    """

    def __init__(self, db_path: str = "shield_memory.db"):
        self.db_path = db_path
        self._manifests: dict[str, AgentManifest] = {}
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── Registration ──────────────────────────────────────────────

    def register(self, manifest: AgentManifest) -> None:
        """Register an agent's capabilities in the mesh."""
        self._manifests[manifest.agent_id] = manifest
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO mesh_manifests 
                   (agent_id, name, capabilities, tools, connectors, trust_score,
                    current_load, max_concurrent, active_tasks, specializations,
                    registered_at, last_heartbeat, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    manifest.agent_id, manifest.name,
                    json.dumps(manifest.capabilities), json.dumps(manifest.tools),
                    json.dumps(manifest.connectors), manifest.trust_score,
                    manifest.current_load, manifest.max_concurrent,
                    manifest.active_tasks, json.dumps(manifest.specializations),
                    manifest.registered_at, manifest.last_heartbeat,
                    json.dumps(manifest.metadata),
                ),
            )
        logger.info(f"Registered agent '{manifest.agent_id}' in mesh (caps={manifest.capabilities})")

    def unregister(self, agent_id: str) -> None:
        self._manifests.pop(agent_id, None)
        with self._conn() as conn:
            conn.execute("DELETE FROM mesh_manifests WHERE agent_id = ?", (agent_id,))

    def heartbeat(self, agent_id: str, load: float = 0.0, active_tasks: int = 0) -> None:
        """Update an agent's liveness and load."""
        if agent_id in self._manifests:
            self._manifests[agent_id].last_heartbeat = time.time()
            self._manifests[agent_id].current_load = load
            self._manifests[agent_id].active_tasks = active_tasks
        with self._conn() as conn:
            conn.execute(
                "UPDATE mesh_manifests SET last_heartbeat = ?, current_load = ?, active_tasks = ? WHERE agent_id = ?",
                (time.time(), load, active_tasks, agent_id),
            )

    # ── Discovery & Matching ──────────────────────────────────────

    async def search(
        self,
        query: str,
        min_trust: float = 0.5,
        max_load: float = 0.9,
        exclude: Optional[list[str]] = None,
        limit: int = 5,
    ) -> list[tuple[float, AgentManifest]]:
        """
        Search the mesh for agents matching a task description.
        
        Uses keyword matching against capabilities, tools, and specializations.
        Returns scored matches sorted by relevance.
        """
        exclude = exclude or []
        all_manifests = self._get_all_manifests()
        query_words = set(query.lower().split())

        scored: list[tuple[float, AgentManifest]] = []

        for manifest in all_manifests:
            if manifest.agent_id in exclude:
                continue
            if manifest.trust_score < min_trust:
                continue
            if manifest.current_load > max_load:
                continue
            # Check if agent has capacity
            if manifest.active_tasks >= manifest.max_concurrent:
                continue
            # Check liveness (consider stale after 5 minutes)
            if time.time() - manifest.last_heartbeat > 300:
                continue

            # Score by capability match
            cap_words = set()
            for cap in manifest.capabilities + manifest.specializations:
                cap_words.update(cap.lower().replace("_", " ").split())
            tool_words = set(t.lower() for t in manifest.tools)
            conn_words = set(c.lower() for c in manifest.connectors)
            
            all_words = cap_words | tool_words | conn_words
            overlap = len(query_words & all_words)
            
            if overlap == 0:
                continue

            # Composite score
            relevance = overlap / max(len(query_words), 1)
            trust_factor = manifest.trust_score
            load_factor = 1 - manifest.current_load
            
            score = relevance * 0.5 + trust_factor * 0.3 + load_factor * 0.2
            scored.append((score, manifest))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    async def request_help(
        self,
        requester: str,
        task: str,
        allowed_tools: Optional[list[str]] = None,
        max_cost: float = 0.0,
        max_time_ms: float = 30000.0,
    ) -> Optional[AgentMatch]:
        """
        Find an agent to help with a task and create a collaboration contract.
        """
        candidates = await self.search(query=task, exclude=[requester])

        if not candidates:
            logger.info(f"No mesh candidates found for task: {task[:60]}")
            return None

        # Try candidates in order of score
        for score, manifest in candidates:
            offer = self._evaluate_offer(manifest, task)
            if offer.accepted:
                contract = CollaborationContract(
                    requester_id=requester,
                    provider_id=manifest.agent_id,
                    task=task,
                    status=ContractStatus.ACTIVE,
                    allowed_tools=allowed_tools or manifest.tools,
                    max_cost=max_cost,
                    max_time_ms=max_time_ms,
                )
                self._save_contract(contract)
                
                # Update provider load
                self.heartbeat(
                    manifest.agent_id,
                    load=min(manifest.current_load + 0.2, 1.0),
                    active_tasks=manifest.active_tasks + 1,
                )

                logger.info(
                    f"Mesh match: {requester} → {manifest.agent_id} "
                    f"for '{task[:40]}' (score={score:.2f})"
                )
                return AgentMatch(
                    provider=manifest,
                    offer=offer,
                    contract=contract,
                    match_score=score,
                )

        logger.info(f"All candidates declined task: {task[:60]}")
        return None

    def _evaluate_offer(self, manifest: AgentManifest, task: str) -> TaskOffer:
        """Evaluate whether an agent can/should accept a task."""
        # Simple acceptance logic (can be made smarter with LLM)
        has_capacity = manifest.active_tasks < manifest.max_concurrent
        has_trust = manifest.trust_score >= 0.5
        
        if has_capacity and has_trust:
            return TaskOffer(
                agent_id=manifest.agent_id,
                accepted=True,
                confidence=manifest.trust_score,
                estimated_time_ms=5000.0,
            )
        else:
            return TaskOffer(
                agent_id=manifest.agent_id,
                accepted=False,
                reason="no capacity" if not has_capacity else "trust too low",
            )

    # ── Contract Management ───────────────────────────────────────

    def _save_contract(self, contract: CollaborationContract):
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO mesh_contracts 
                   (id, requester_id, provider_id, task, status, data_sharing,
                    allowed_tools, forbidden_tools, max_cost, max_time_ms,
                    result, error, actual_cost, actual_time_ms, created_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    contract.id, contract.requester_id, contract.provider_id,
                    contract.task, contract.status.value,
                    json.dumps(contract.data_sharing), json.dumps(contract.allowed_tools),
                    json.dumps(contract.forbidden_tools), contract.max_cost,
                    contract.max_time_ms, json.dumps(contract.result, default=str) if contract.result else None,
                    contract.error, contract.actual_cost, contract.actual_time_ms,
                    contract.created_at, contract.completed_at,
                ),
            )

    async def complete_contract(
        self,
        contract_id: str,
        result: Any = None,
        error: Optional[str] = None,
        actual_cost: float = 0.0,
        actual_time_ms: float = 0.0,
    ) -> None:
        """Mark a collaboration contract as completed."""
        status = ContractStatus.COMPLETED if not error else ContractStatus.FAILED
        with self._conn() as conn:
            conn.execute(
                """UPDATE mesh_contracts 
                   SET status = ?, result = ?, error = ?, actual_cost = ?, 
                       actual_time_ms = ?, completed_at = ?
                   WHERE id = ?""",
                (status.value, json.dumps(result, default=str), error,
                 actual_cost, actual_time_ms, time.time(), contract_id),
            )
        logger.info(f"Contract {contract_id} → {status.value}")

    async def get_active_contracts(self, agent_id: Optional[str] = None) -> list[CollaborationContract]:
        with self._conn() as conn:
            if agent_id:
                rows = conn.execute(
                    """SELECT * FROM mesh_contracts 
                       WHERE status = 'active' AND (requester_id = ? OR provider_id = ?)""",
                    (agent_id, agent_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM mesh_contracts WHERE status = 'active'"
                ).fetchall()
        return [self._row_to_contract(r) for r in rows]

    # ── Stats & Querying ──────────────────────────────────────────

    def _get_all_manifests(self) -> list[AgentManifest]:
        """Get all manifests (in-memory + DB)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM mesh_manifests").fetchall()
        manifests = {}
        for r in rows:
            m = AgentManifest(
                agent_id=r[0], name=r[1],
                capabilities=json.loads(r[2] or "[]"),
                tools=json.loads(r[3] or "[]"),
                connectors=json.loads(r[4] or "[]"),
                trust_score=r[5], current_load=r[6],
                max_concurrent=r[7], active_tasks=r[8],
                specializations=json.loads(r[9] or "[]"),
                registered_at=r[10], last_heartbeat=r[11],
                metadata=json.loads(r[12] or "{}"),
            )
            manifests[m.agent_id] = m
        # Overlay in-memory updates
        manifests.update(self._manifests)
        return list(manifests.values())

    async def get_mesh_topology(self) -> dict:
        """Get the current mesh state for visualization."""
        manifests = self._get_all_manifests()
        with self._conn() as conn:
            active = conn.execute("SELECT COUNT(*) FROM mesh_contracts WHERE status = 'active'").fetchone()[0]
            completed = conn.execute("SELECT COUNT(*) FROM mesh_contracts WHERE status = 'completed'").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM mesh_contracts WHERE status = 'failed'").fetchone()[0]
        
        return {
            "agents": [
                {
                    "id": m.agent_id,
                    "name": m.name,
                    "capabilities": m.capabilities,
                    "tools": m.tools,
                    "trust": m.trust_score,
                    "load": m.current_load,
                    "active_tasks": m.active_tasks,
                    "alive": time.time() - m.last_heartbeat < 300,
                }
                for m in manifests
            ],
            "contracts": {
                "active": active,
                "completed": completed,
                "failed": failed,
            },
            "total_agents": len(manifests),
        }

    async def get_stats(self) -> dict:
        topology = await self.get_mesh_topology()
        alive = sum(1 for a in topology["agents"] if a["alive"])
        return {
            "total_agents": topology["total_agents"],
            "alive_agents": alive,
            "active_contracts": topology["contracts"]["active"],
            "completed_contracts": topology["contracts"]["completed"],
            "failed_contracts": topology["contracts"]["failed"],
        }

    def _row_to_contract(self, row: tuple) -> CollaborationContract:
        return CollaborationContract(
            id=row[0], requester_id=row[1], provider_id=row[2],
            task=row[3], status=ContractStatus(row[4]),
            data_sharing=json.loads(row[5] or "[]"),
            allowed_tools=json.loads(row[6] or "[]"),
            forbidden_tools=json.loads(row[7] or "[]"),
            max_cost=row[8], max_time_ms=row[9],
            result=json.loads(row[10]) if row[10] else None,
            error=row[11], actual_cost=row[12],
            actual_time_ms=row[13], created_at=row[14],
            completed_at=row[15],
        )
