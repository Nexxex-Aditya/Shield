"""
Shield Command — Semantic Cache

2-tier intelligent caching for LLM responses.

L1: ExactCache — LRU dictionary, O(1) lookup, TTL-based expiry
L2: NearMatchCache — Uses vector similarity to find semantically
    similar past queries (threshold: 0.92) and return cached responses

Saves money by not re-calling the LLM for similar questions.

Integration points:
    - EmbeddingEngine: embeds queries for L2 similarity search
    - VectorStore: stores query embeddings for near-match lookup
    - ModelRegistry: wraps LLM calls with cache-first logic
"""

import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import OrderedDict

logger = logging.getLogger("shield.semantic_cache")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A cached query-response pair."""
    query: str
    response: Any
    model: str = ""
    embedding: list[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600  # 1 hour default
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_queries: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    misses: int = 0
    entries: int = 0
    estimated_savings_usd: float = 0.0
    avg_l1_latency_ms: float = 0.0
    avg_l2_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        return (self.l1_hits + self.l2_hits) / max(self.total_queries, 1)


# ---------------------------------------------------------------------------
# L1: Exact Match Cache (LRU)
# ---------------------------------------------------------------------------

class ExactCache:
    """
    Fast LRU cache for exact query matches.
    O(1) lookup using hash keys.
    """

    def __init__(self, max_size: int = 5000, default_ttl: float = 3600):
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl

    def _key(self, query: str, model: str = "") -> str:
        return hashlib.sha256(f"{model}:{query}".encode()).hexdigest()

    def get(self, query: str, model: str = "") -> Optional[CacheEntry]:
        key = self._key(query, model)
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._store[key]
            return None
        entry.hit_count += 1
        self._store.move_to_end(key)
        return entry

    def put(self, query: str, response: Any, model: str = "", ttl: float = 0):
        key = self._key(query, model)
        entry = CacheEntry(
            query=query,
            response=response,
            model=model,
            ttl_seconds=ttl or self.default_ttl,
        )
        self._store[key] = entry
        self._store.move_to_end(key)
        # Evict oldest if over capacity
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self):
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    def evict_expired(self) -> int:
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
        return len(expired)


# ---------------------------------------------------------------------------
# L2: Semantic Near-Match Cache
# ---------------------------------------------------------------------------

class NearMatchCache:
    """
    Finds semantically similar past queries using vector similarity.
    If a past query is >92% similar, returns the cached response.
    """

    COLLECTION = "semantic_cache"

    def __init__(
        self,
        vector_store=None,
        embedding_engine=None,
        similarity_threshold: float = 0.92,
        max_entries: int = 10000,
    ):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self._response_map: dict[str, CacheEntry] = {}  # doc_id → CacheEntry

    async def get(self, query: str, model: str = "") -> Optional[CacheEntry]:
        """Find a semantically similar cached response."""
        if not self.vector_store or not self.embedding_engine:
            return None

        # Embed the query
        embed_result = await self.embedding_engine.embed(query)

        # Search for similar past queries
        results = await self.vector_store.search(
            query_embedding=embed_result.vector,
            collection=self.COLLECTION,
            top_k=3,
            min_score=self.threshold,
        )

        if not results:
            return None

        # Return the best match
        best = results[0]
        entry = self._response_map.get(best.id)
        if entry and not entry.is_expired:
            entry.hit_count += 1
            logger.debug(
                f"Semantic cache hit: score={best.score:.3f} "
                f"query='{query[:50]}' matched='{entry.query[:50]}'"
            )
            return entry

        return None

    async def put(self, query: str, response: Any, model: str = "", ttl: float = 3600):
        """Store a query-response pair for future near-match lookups."""
        if not self.vector_store or not self.embedding_engine:
            return

        embed_result = await self.embedding_engine.embed(query)

        from .vector_store import VectorDocument
        doc = VectorDocument(
            text=query,
            embedding=embed_result.vector,
            collection=self.COLLECTION,
            metadata={"model": model},
        )

        await self.vector_store.add(doc)
        self._response_map[doc.id] = CacheEntry(
            query=query,
            response=response,
            model=model,
            embedding=embed_result.vector,
            ttl_seconds=ttl,
        )

    @property
    def size(self) -> int:
        return len(self._response_map)


# ---------------------------------------------------------------------------
# Unified Semantic Cache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    2-tier caching: L1 exact match + L2 semantic near-match.
    
    Usage:
        cache = SemanticCache(
            vector_store=store,
            embedding_engine=engine,
        )
        
        # Check cache before calling LLM
        cached = await cache.get("What is Shield?", model="gpt-4")
        if cached:
            return cached.response  # Free!
        
        # Cache miss — call LLM
        response = await llm.generate("What is Shield?")
        
        # Store for future use
        await cache.put("What is Shield?", response, model="gpt-4")
    """

    # Estimated cost per LLM call (conservative)
    COST_PER_CALL = 0.003

    def __init__(
        self,
        vector_store=None,
        embedding_engine=None,
        l1_max_size: int = 5000,
        l1_ttl: float = 3600,
        l2_threshold: float = 0.92,
    ):
        self.l1 = ExactCache(max_size=l1_max_size, default_ttl=l1_ttl)
        self.l2 = NearMatchCache(
            vector_store=vector_store,
            embedding_engine=embedding_engine,
            similarity_threshold=l2_threshold,
        )
        self._stats = CacheStats()

    async def get(self, query: str, model: str = "") -> Optional[CacheEntry]:
        """
        Check cache. L1 (exact) first, then L2 (semantic).
        """
        self._stats.total_queries += 1

        # L1: exact match
        start = time.time()
        entry = self.l1.get(query, model)
        if entry:
            self._stats.l1_hits += 1
            self._stats.estimated_savings_usd += self.COST_PER_CALL
            l1_time = (time.time() - start) * 1000
            self._stats.avg_l1_latency_ms = (
                (self._stats.avg_l1_latency_ms * (self._stats.l1_hits - 1) + l1_time)
                / self._stats.l1_hits
            )
            return entry

        # L2: semantic near-match
        start = time.time()
        entry = await self.l2.get(query, model)
        if entry:
            self._stats.l2_hits += 1
            self._stats.estimated_savings_usd += self.COST_PER_CALL
            l2_time = (time.time() - start) * 1000
            self._stats.avg_l2_latency_ms = (
                (self._stats.avg_l2_latency_ms * (self._stats.l2_hits - 1) + l2_time)
                / self._stats.l2_hits
            )
            # Promote to L1 for faster future lookups
            self.l1.put(query, entry.response, model)
            return entry

        self._stats.misses += 1
        return None

    async def put(self, query: str, response: Any, model: str = "", ttl: float = 3600):
        """Store in both L1 and L2."""
        self.l1.put(query, response, model, ttl)
        await self.l2.put(query, response, model, ttl)

    def clear(self):
        """Clear all caches."""
        self.l1.clear()

    async def get_stats(self) -> dict:
        self._stats.entries = self.l1.size + self.l2.size
        return {
            "total_queries": self._stats.total_queries,
            "l1_hits": self._stats.l1_hits,
            "l2_hits": self._stats.l2_hits,
            "misses": self._stats.misses,
            "hit_rate": round(self._stats.hit_rate, 3),
            "entries": self._stats.entries,
            "l1_entries": self.l1.size,
            "l2_entries": self.l2.size,
            "estimated_savings_usd": round(self._stats.estimated_savings_usd, 4),
            "avg_l1_latency_ms": round(self._stats.avg_l1_latency_ms, 2),
            "avg_l2_latency_ms": round(self._stats.avg_l2_latency_ms, 2),
        }
