"""
Shield Command — Cognitive Memory Architecture (CMA)

Three-layer memory system inspired by cognitive science:

    Working Memory:  Current task context. Fast, volatile, cleared per run.
    Episodic Memory: Past experiences with outcomes. Similarity-searchable.
                     Decays over time via forgetting curve.
    Semantic Memory: Permanent distilled knowledge. Built by consolidating
                     clusters of episodic memories into general rules.

Integration points:
    - CognitiveGraph uses CMA to inform agent decisions
    - KnowledgeBroker reads/writes to SemanticMemory for cross-agent propagation
    - AgentGenetics accesses episodic results for fitness evaluation
"""

import json
import math
import time
import uuid
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("shield.cognitive_memory")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    context: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    impact_score: float = 0.5       # 0=irrelevant, 1=critical
    emotional_tag: str = "neutral"   # success / failure / surprise / neutral
    source_agent: str = ""
    source_run_id: str = ""


@dataclass
class SemanticRule:
    """A distilled knowledge rule extracted from episodic clusters."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule: str = ""                   # The principle: "Stripe requires idempotency keys"
    confidence: float = 0.8
    evidence_count: int = 1          # How many episodes support this
    related_tools: list[str] = field(default_factory=list)
    related_connectors: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    source: str = "self-learned"     # or "propagated:agent-xyz"


@dataclass
class MemoryBundle:
    """Combined retrieval result from all memory layers."""
    working: dict = field(default_factory=dict)
    episodes: list[MemoryEntry] = field(default_factory=list)
    rules: list[SemanticRule] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Working Memory — Fast volatile context for current run
# ---------------------------------------------------------------------------

class WorkingMemory:
    """
    In-process scratchpad for the current agent run.
    Thread-safe via copy-on-read. Cleared after each execution.
    """

    def __init__(self):
        self._context: dict[str, Any] = {}
        self._stack: list[dict] = []

    def set(self, key: str, value: Any) -> None:
        self._context[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def get_context(self) -> dict:
        return dict(self._context)

    def push_frame(self) -> None:
        """Save current context (for sub-goal execution)."""
        self._stack.append(dict(self._context))

    def pop_frame(self) -> dict:
        """Restore previous context frame."""
        if self._stack:
            self._context = self._stack.pop()
        return dict(self._context)

    def clear(self) -> None:
        self._context.clear()
        self._stack.clear()

    def __repr__(self) -> str:
        return f"WorkingMemory(keys={list(self._context.keys())}, depth={len(self._stack)})"


# ---------------------------------------------------------------------------
# Episodic Memory — Experience store with forgetting curve
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    SQLite-backed store of past experiences.
    
    Each episode is a record of: what happened, what the context was,
    whether it succeeded or failed, and how impactful it was.
    
    Retrieval uses a composite score:
        score = (similarity_weight * text_match) 
              + (recency_weight * recency_decay) 
              + (impact_weight * impact_score)
    
    Forgetting curve: memories decay exponentially based on age and access.
    The half-life is configurable (default 7 days for low-impact memories).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        context TEXT DEFAULT '{}',
        tags TEXT DEFAULT '[]',
        created_at REAL NOT NULL,
        last_accessed REAL NOT NULL,
        access_count INTEGER DEFAULT 0,
        impact_score REAL DEFAULT 0.5,
        emotional_tag TEXT DEFAULT 'neutral',
        source_agent TEXT DEFAULT '',
        source_run_id TEXT DEFAULT '',
        consolidated INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(source_agent);
    CREATE INDEX IF NOT EXISTS idx_episodes_emotional ON episodes(emotional_tag);
    CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated);
    """

    def __init__(self, db_path: str = "shield_memory.db", half_life_days: float = 7.0):
        self.db_path = db_path
        self.half_life = half_life_days * 86400  # convert to seconds
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    async def store(self, entry: MemoryEntry) -> str:
        """Store a new episodic memory."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO episodes 
                   (id, content, context, tags, created_at, last_accessed,
                    access_count, impact_score, emotional_tag, source_agent, source_run_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.id, entry.content, json.dumps(entry.context),
                    json.dumps(entry.tags), entry.created_at, entry.last_accessed,
                    entry.access_count, entry.impact_score, entry.emotional_tag,
                    entry.source_agent, entry.source_run_id,
                ),
            )
        logger.info(f"Stored episode {entry.id[:8]} [{entry.emotional_tag}] impact={entry.impact_score:.2f}")
        return entry.id

    async def search(
        self,
        query: str,
        recency_weight: float = 0.3,
        impact_weight: float = 0.5,
        similarity_weight: float = 0.2,
        limit: int = 10,
        agent_filter: Optional[str] = None,
    ) -> list[MemoryEntry]:
        """
        Search episodic memories with composite scoring.
        
        Uses keyword matching for similarity (production would use embeddings).
        Applies forgetting curve to penalize old, low-impact memories.
        """
        now = time.time()
        with self._conn() as conn:
            condition = "WHERE 1=1"
            params: list[Any] = []
            if agent_filter:
                condition += " AND source_agent = ?"
                params.append(agent_filter)

            rows = conn.execute(
                f"SELECT * FROM episodes {condition} ORDER BY created_at DESC LIMIT 500",
                params,
            ).fetchall()

        entries: list[tuple[float, MemoryEntry]] = []
        query_words = set(query.lower().split())

        for row in rows:
            entry = self._row_to_entry(row)
            
            # Similarity: keyword overlap (placeholder for embedding search)
            content_words = set(entry.content.lower().split())
            tag_words = set(t.lower() for t in entry.tags)
            overlap = len(query_words & (content_words | tag_words))
            sim_score = min(overlap / max(len(query_words), 1), 1.0)

            # Recency: exponential decay (forgetting curve)
            age = now - entry.created_at
            effective_half_life = self.half_life * (1 + entry.impact_score)  # high-impact decays slower
            recency_score = math.exp(-0.693 * age / max(effective_half_life, 1))

            # Composite score
            score = (
                similarity_weight * sim_score
                + recency_weight * recency_score
                + impact_weight * entry.impact_score
            )
            entries.append((score, entry))

        # Sort by score descending, return top N
        entries.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in entries[:limit]]

        # Update access timestamps
        if results:
            with self._conn() as conn:
                for entry in results:
                    conn.execute(
                        "UPDATE episodes SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                        (now, entry.id),
                    )

        return results

    async def get_unconsolidated(self, limit: int = 200) -> list[MemoryEntry]:
        """Get episodes that haven't been consolidated into semantic memory yet."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE consolidated = 0 ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    async def mark_consolidated(self, entry_ids: list[str]) -> None:
        """Mark episodes as consolidated."""
        with self._conn() as conn:
            conn.executemany(
                "UPDATE episodes SET consolidated = 1 WHERE id = ?",
                [(eid,) for eid in entry_ids],
            )

    async def prune_expired(self, min_score: float = 0.01) -> int:
        """Remove memories that have decayed below the minimum relevance threshold."""
        now = time.time()
        with self._conn() as conn:
            rows = conn.execute("SELECT id, created_at, impact_score FROM episodes").fetchall()
            to_delete = []
            for row in rows:
                age = now - row[1]
                effective_half_life = self.half_life * (1 + row[2])
                recency_score = math.exp(-0.693 * age / max(effective_half_life, 1))
                if recency_score * row[2] < min_score:
                    to_delete.append((row[0],))
            if to_delete:
                conn.executemany("DELETE FROM episodes WHERE id = ?", to_delete)
        logger.info(f"Pruned {len(to_delete)} expired episodic memories")
        return len(to_delete)

    async def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def _row_to_entry(self, row: tuple) -> MemoryEntry:
        return MemoryEntry(
            id=row[0], content=row[1], context=json.loads(row[2] or "{}"),
            tags=json.loads(row[3] or "[]"), created_at=row[4], last_accessed=row[5],
            access_count=row[6], impact_score=row[7], emotional_tag=row[8],
            source_agent=row[9], source_run_id=row[10],
        )


# ---------------------------------------------------------------------------
# Semantic Memory — Permanent distilled knowledge
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Permanent store of learned rules and principles.
    
    Rules are distilled from clusters of similar episodic memories.
    Example: "Stripe API always requires idempotency keys for POST requests"
    — learned from 5 separate failure episodes involving Stripe.
    
    Rules can also be propagated from other agents via KnowledgeBroker.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS semantic_rules (
        id TEXT PRIMARY KEY,
        rule TEXT NOT NULL,
        confidence REAL DEFAULT 0.8,
        evidence_count INTEGER DEFAULT 1,
        related_tools TEXT DEFAULT '[]',
        related_connectors TEXT DEFAULT '[]',
        created_at REAL NOT NULL,
        source TEXT DEFAULT 'self-learned'
    );
    CREATE INDEX IF NOT EXISTS idx_rules_confidence ON semantic_rules(confidence);
    """

    def __init__(self, db_path: str = "shield_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    async def store(self, rule: SemanticRule) -> str:
        """Store or update a semantic rule."""
        with self._conn() as conn:
            # Check for duplicate rules (similar content)
            existing = conn.execute(
                "SELECT id, evidence_count, confidence FROM semantic_rules WHERE rule = ?",
                (rule.rule,),
            ).fetchone()

            if existing:
                # Reinforce existing rule
                conn.execute(
                    """UPDATE semantic_rules 
                       SET evidence_count = evidence_count + 1, 
                           confidence = MIN(confidence + 0.05, 1.0)
                       WHERE id = ?""",
                    (existing[0],),
                )
                logger.info(f"Reinforced rule {existing[0][:8]} (evidence={existing[1]+1})")
                return existing[0]
            else:
                conn.execute(
                    """INSERT INTO semantic_rules 
                       (id, rule, confidence, evidence_count, related_tools, 
                        related_connectors, created_at, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        rule.id, rule.rule, rule.confidence, rule.evidence_count,
                        json.dumps(rule.related_tools), json.dumps(rule.related_connectors),
                        rule.created_at, rule.source,
                    ),
                )
                logger.info(f"Stored new rule {rule.id[:8]}: {rule.rule[:60]}")
                return rule.id

    async def search(self, query: str, min_confidence: float = 0.3, limit: int = 10) -> list[SemanticRule]:
        """Search semantic rules by keyword relevance."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM semantic_rules 
                   WHERE confidence >= ? 
                   ORDER BY confidence DESC, evidence_count DESC 
                   LIMIT ?""",
                (min_confidence, limit * 5),
            ).fetchall()

        query_words = set(query.lower().split())
        scored: list[tuple[float, SemanticRule]] = []

        for row in rows:
            rule = self._row_to_rule(row)
            rule_words = set(rule.rule.lower().split())
            tool_words = set(t.lower() for t in rule.related_tools)
            conn_words = set(c.lower() for c in rule.related_connectors)
            all_words = rule_words | tool_words | conn_words

            overlap = len(query_words & all_words)
            sim = overlap / max(len(query_words), 1)
            score = sim * 0.6 + rule.confidence * 0.3 + min(rule.evidence_count / 10, 1) * 0.1
            if score > 0.05:
                scored.append((score, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    async def get_all(self, min_confidence: float = 0.0) -> list[SemanticRule]:
        """Get all semantic rules above a confidence threshold."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_rules WHERE confidence >= ? ORDER BY confidence DESC",
                (min_confidence,),
            ).fetchall()
        return [self._row_to_rule(row) for row in rows]

    async def get_rules_for_tools(self, tool_names: list[str]) -> list[SemanticRule]:
        """Get rules relevant to specific tools."""
        all_rules = await self.get_all(min_confidence=0.3)
        return [
            r for r in all_rules
            if any(t in r.related_tools for t in tool_names)
        ]

    async def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM semantic_rules").fetchone()[0]

    def _row_to_rule(self, row: tuple) -> SemanticRule:
        return SemanticRule(
            id=row[0], rule=row[1], confidence=row[2], evidence_count=row[3],
            related_tools=json.loads(row[4] or "[]"),
            related_connectors=json.loads(row[5] or "[]"),
            created_at=row[6], source=row[7],
        )


# ---------------------------------------------------------------------------
# Cognitive Memory Manager — Unified interface
# ---------------------------------------------------------------------------

class CognitiveMemoryManager:
    """
    Unified interface to the three-layer memory system.
    
    Usage:
        memory = CognitiveMemoryManager()
        
        # During a run — store working context
        memory.working.set("current_goal", "Analyze revenue")
        
        # After a run — store the experience
        await memory.record_episode(
            content="Called Stripe API, got 403 — missing idempotency key",
            tags=["stripe", "api", "error"],
            impact_score=0.8,
            emotional_tag="failure",
            agent_id="agent-revenue",
        )
        
        # Before a run — recall relevant memories
        bundle = await memory.remember("stripe payment processing")
        # bundle.working = current context
        # bundle.episodes = relevant past experiences
        # bundle.rules = learned principles
        
        # Nightly — consolidate episodes into rules
        await memory.consolidate(llm_adapter=adapter)
    """

    def __init__(self, db_path: str = "shield_memory.db", half_life_days: float = 7.0):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(db_path=db_path, half_life_days=half_life_days)
        self.semantic = SemanticMemory(db_path=db_path)
        self.db_path = db_path

    async def remember(
        self,
        query: str,
        recency_weight: float = 0.3,
        impact_weight: float = 0.5,
        similarity_weight: float = 0.2,
        agent_filter: Optional[str] = None,
    ) -> MemoryBundle:
        """Retrieve relevant memories from all three layers."""
        episodes = await self.episodic.search(
            query=query,
            recency_weight=recency_weight,
            impact_weight=impact_weight,
            similarity_weight=similarity_weight,
            agent_filter=agent_filter,
        )
        rules = await self.semantic.search(query)
        working = self.working.get_context()

        return MemoryBundle(working=working, episodes=episodes, rules=rules)

    async def record_episode(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        context: Optional[dict] = None,
        impact_score: float = 0.5,
        emotional_tag: str = "neutral",
        agent_id: str = "",
        run_id: str = "",
    ) -> str:
        """Record a new experience."""
        entry = MemoryEntry(
            content=content,
            tags=tags or [],
            context=context or {},
            impact_score=impact_score,
            emotional_tag=emotional_tag,
            source_agent=agent_id,
            source_run_id=run_id,
        )
        return await self.episodic.store(entry)

    async def consolidate(self, llm_adapter=None) -> list[SemanticRule]:
        """
        Consolidate unconsolidated episodic memories into semantic rules.
        
        Groups similar episodes, uses LLM (if available) to extract
        general principles, and stores them as semantic rules.
        """
        episodes = await self.episodic.get_unconsolidated()
        if not episodes:
            return []

        # Group episodes by tags
        clusters: dict[str, list[MemoryEntry]] = {}
        for ep in episodes:
            key = "|".join(sorted(ep.tags[:3])) if ep.tags else "general"
            clusters.setdefault(key, []).append(ep)

        new_rules: list[SemanticRule] = []

        for cluster_key, cluster_episodes in clusters.items():
            if len(cluster_episodes) < 2:
                continue  # Need at least 2 episodes to form a rule

            # Extract tools and connectors from the cluster
            tools = set()
            connectors = set()
            for ep in cluster_episodes:
                tools.update(ep.tags)
                if ep.context.get("connector"):
                    connectors.add(ep.context["connector"])

            if llm_adapter:
                # Use LLM to extract a principle from the cluster
                episode_summaries = "\n".join(
                    f"- [{ep.emotional_tag}] {ep.content}" for ep in cluster_episodes[:10]
                )
                try:
                    response = await llm_adapter.generate(
                        system_prompt=(
                            "You are a knowledge distillation system. Given a cluster of "
                            "past experiences, extract ONE concise, actionable rule or principle "
                            "that would help an agent avoid failures or replicate successes. "
                            "Respond with ONLY the rule, one sentence."
                        ),
                        user_prompt=f"Experiences:\n{episode_summaries}",
                        temperature=0.3,
                        max_tokens=100,
                    )
                    rule_text = response.get("content", "").strip()
                except Exception as e:
                    logger.warning(f"LLM consolidation failed: {e}")
                    rule_text = ""
            else:
                # Fallback: extract common pattern from failure episodes
                failures = [ep for ep in cluster_episodes if ep.emotional_tag == "failure"]
                if failures:
                    rule_text = f"Caution with {cluster_key}: {len(failures)}/{len(cluster_episodes)} attempts failed"
                else:
                    rule_text = f"Pattern observed in {cluster_key}: {len(cluster_episodes)} consistent results"

            if rule_text:
                rule = SemanticRule(
                    rule=rule_text,
                    confidence=min(0.5 + len(cluster_episodes) * 0.1, 0.95),
                    evidence_count=len(cluster_episodes),
                    related_tools=list(tools)[:10],
                    related_connectors=list(connectors)[:5],
                )
                await self.semantic.store(rule)
                new_rules.append(rule)

        # Mark all episodes as consolidated
        all_ids = [ep.id for ep in episodes]
        await self.episodic.mark_consolidated(all_ids)

        logger.info(f"Consolidated {len(episodes)} episodes into {len(new_rules)} rules")
        return new_rules

    async def get_stats(self) -> dict:
        """Get memory system statistics."""
        return {
            "working_memory_keys": len(self.working.get_context()),
            "episodic_count": await self.episodic.count(),
            "semantic_rules_count": await self.semantic.count(),
        }

    def reset_working(self) -> None:
        """Clear working memory for a new run."""
        self.working.clear()
