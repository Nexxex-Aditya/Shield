"""
Shield Command — Adaptive Retrieval Cortex

RAG engine that LEARNS which retrieval strategy works best
for each query type. Dynamically switches between strategies
and self-improves over time.

Strategies:
    1. DENSE — Pure vector similarity search
    2. SPARSE_BM25 — Keyword-based BM25 scoring
    3. HYBRID — Combined dense + sparse
    4. RERANKED — Dense retrieval + cross-encoder reranking

Integration points:
    - VectorStore: dense retrieval backend
    - EmbeddingEngine: embeds queries + reranks
    - SemanticCache: caches retrieval results
    - DocumentProcessor: ingests source documents
    - CognitiveMemory: learns from retrieval feedback
"""

import time
import math
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import Counter, defaultdict
from enum import Enum

logger = logging.getLogger("shield.retrieval_cortex")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    SPARSE_BM25 = "sparse_bm25"
    HYBRID = "hybrid"
    RERANKED = "reranked"


@dataclass
class RetrievalResult:
    """A single retrieved chunk."""
    text: str = ""
    score: float = 0.0
    source: str = ""
    metadata: dict = field(default_factory=dict)
    strategy: str = ""
    chunk_id: str = ""


@dataclass
class RetrievalResponse:
    """Complete retrieval response."""
    query: str = ""
    results: list[RetrievalResult] = field(default_factory=list)
    strategy_used: RetrievalStrategy = RetrievalStrategy.DENSE
    total_candidates: int = 0
    duration_ms: float = 0.0
    from_cache: bool = False


@dataclass
class StrategyPerformance:
    """Tracks how well each strategy performs."""
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE
    total_queries: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.positive_feedback + self.negative_feedback
        return self.positive_feedback / max(total, 1)


# ---------------------------------------------------------------------------
# Sparse BM25 Retriever
# ---------------------------------------------------------------------------

class SparseBM25Retriever:
    """
    Pure-Python BM25 scoring for keyword-based retrieval.
    No external dependencies.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: list[dict] = []  # {id, text, tokens, metadata}
        self._df: Counter = Counter()  # document frequency
        self._avg_dl: float = 0.0
        self._n: int = 0

    def add_documents(self, documents: list[dict]):
        """Add documents for BM25 indexing."""
        for doc in documents:
            tokens = self._tokenize(doc.get("text", ""))
            doc["tokens"] = tokens
            self._documents.append(doc)
            for token in set(tokens):
                self._df[token] += 1

        self._n = len(self._documents)
        if self._n > 0:
            self._avg_dl = sum(len(d["tokens"]) for d in self._documents) / self._n

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search using BM25 scoring."""
        query_tokens = self._tokenize(query)
        scores = []

        for doc in self._documents:
            score = self._bm25_score(query_tokens, doc["tokens"])
            if score > 0:
                scores.append({
                    "id": doc.get("id", ""),
                    "text": doc.get("text", ""),
                    "score": score,
                    "metadata": doc.get("metadata", {}),
                })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        dl = len(doc_tokens)
        score = 0.0
        doc_tf = Counter(doc_tokens)

        for token in query_tokens:
            if token not in doc_tf:
                continue
            tf = doc_tf[token]
            df = self._df.get(token, 0)
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
            score += idf * numerator / denominator

        return score

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())


# ---------------------------------------------------------------------------
# Strategy Selector
# ---------------------------------------------------------------------------

class StrategySelector:
    """
    Learns which retrieval strategy works best for each query type.
    Uses a multi-armed bandit approach (epsilon-greedy).
    """

    def __init__(self, epsilon: float = 0.15):
        self.epsilon = epsilon
        self._performance: dict[str, StrategyPerformance] = {}
        for strategy in RetrievalStrategy:
            self._performance[strategy.value] = StrategyPerformance(strategy=strategy)

    def select(self, query: str) -> RetrievalStrategy:
        """Select the best strategy for a query."""
        import random

        # Exploration: try random strategy
        if random.random() < self.epsilon:
            return random.choice(list(RetrievalStrategy))

        # Exploitation: pick the best performing strategy
        best = max(
            self._performance.values(),
            key=lambda p: p.success_rate * 0.7 + (1 / max(p.avg_latency_ms, 1)) * 0.3,
        )
        return best.strategy

    def record_feedback(self, strategy: RetrievalStrategy, positive: bool, latency_ms: float = 0):
        """Record feedback on a strategy's performance."""
        perf = self._performance[strategy.value]
        perf.total_queries += 1
        if positive:
            perf.positive_feedback += 1
        else:
            perf.negative_feedback += 1
        perf.avg_latency_ms = (
            (perf.avg_latency_ms * (perf.total_queries - 1) + latency_ms)
            / perf.total_queries
        )

    def get_stats(self) -> dict:
        return {
            s.value: {
                "queries": self._performance[s.value].total_queries,
                "success_rate": round(self._performance[s.value].success_rate, 3),
                "avg_latency_ms": round(self._performance[s.value].avg_latency_ms, 1),
            }
            for s in RetrievalStrategy
        }


# ---------------------------------------------------------------------------
# Retrieval Cortex
# ---------------------------------------------------------------------------

class RetrievalCortex:
    """
    Adaptive RAG engine that learns which strategy works best.
    
    Usage:
        cortex = RetrievalCortex(
            embedding_engine=engine,
            vector_store=store,
        )
        
        # Retrieve
        response = await cortex.retrieve("What is Shield's security model?")
        for r in response.results:
            print(f"  [{r.score:.3f}] {r.text[:80]}")
        
        # Provide feedback to help it learn
        await cortex.feedback(response, positive=True)
    """

    def __init__(
        self,
        embedding_engine=None,
        vector_store=None,
        semantic_cache=None,
        top_k: int = 5,
        collection: str = "documents",
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.semantic_cache = semantic_cache
        self.top_k = top_k
        self.collection = collection
        self.selector = StrategySelector()
        self._bm25 = SparseBM25Retriever()
        self._bm25_loaded = False

        # Stats
        self.total_queries = 0
        self.cache_hits = 0

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        strategy: RetrievalStrategy = None,
        collection: str = None,
    ) -> RetrievalResponse:
        """
        Retrieve relevant documents for a query.
        Auto-selects the best strategy unless one is specified.
        """
        start_time = time.time()
        self.total_queries += 1
        top_k = top_k or self.top_k
        collection = collection or self.collection

        # Check cache first
        if self.semantic_cache:
            cached = await self.semantic_cache.get(f"retrieval:{query}")
            if cached:
                self.cache_hits += 1
                return RetrievalResponse(
                    query=query,
                    results=cached.response,
                    from_cache=True,
                    duration_ms=(time.time() - start_time) * 1000,
                )

        # Select strategy
        selected = strategy or self.selector.select(query)

        # Execute retrieval
        if selected == RetrievalStrategy.DENSE:
            results = await self._dense_retrieve(query, top_k, collection)
        elif selected == RetrievalStrategy.SPARSE_BM25:
            results = await self._sparse_retrieve(query, top_k)
        elif selected == RetrievalStrategy.HYBRID:
            results = await self._hybrid_retrieve(query, top_k, collection)
        elif selected == RetrievalStrategy.RERANKED:
            results = await self._reranked_retrieve(query, top_k, collection)
        else:
            results = await self._dense_retrieve(query, top_k, collection)

        for r in results:
            r.strategy = selected.value

        duration = (time.time() - start_time) * 1000
        response = RetrievalResponse(
            query=query,
            results=results,
            strategy_used=selected,
            total_candidates=len(results),
            duration_ms=duration,
        )

        # Cache the results
        if self.semantic_cache and results:
            await self.semantic_cache.put(f"retrieval:{query}", results)

        logger.debug(
            f"Retrieved {len(results)} results for '{query[:50]}' "
            f"strategy={selected.value} duration={duration:.0f}ms"
        )
        return response

    async def feedback(self, response: RetrievalResponse, positive: bool):
        """Provide feedback on retrieval quality to help the system learn."""
        self.selector.record_feedback(
            response.strategy_used,
            positive=positive,
            latency_ms=response.duration_ms,
        )

    # ── Strategy Implementations ──────────────────────────────────

    async def _dense_retrieve(self, query: str, top_k: int, collection: str) -> list[RetrievalResult]:
        """Pure vector similarity search."""
        if not self.embedding_engine or not self.vector_store:
            return []

        embed = await self.embedding_engine.embed(query)
        results = await self.vector_store.search(
            query_embedding=embed.vector,
            collection=collection,
            top_k=top_k,
        )

        return [
            RetrievalResult(
                text=r.text,
                score=r.score,
                source=r.metadata.get("source_name", ""),
                metadata=r.metadata,
                chunk_id=r.id,
            )
            for r in results
        ]

    async def _sparse_retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """BM25 keyword-based retrieval."""
        results = self._bm25.search(query, top_k)
        return [
            RetrievalResult(
                text=r["text"],
                score=r["score"],
                metadata=r.get("metadata", {}),
                chunk_id=r.get("id", ""),
            )
            for r in results
        ]

    async def _hybrid_retrieve(self, query: str, top_k: int, collection: str) -> list[RetrievalResult]:
        """Combined dense + sparse with reciprocal rank fusion."""
        dense_results = await self._dense_retrieve(query, top_k * 2, collection)
        sparse_results = await self._sparse_retrieve(query, top_k * 2)

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, RetrievalResult] = {}
        k = 60  # RRF constant

        for rank, r in enumerate(dense_results):
            key = r.chunk_id or r.text[:100]
            rrf_scores[key] += 1 / (k + rank + 1)
            doc_map[key] = r

        for rank, r in enumerate(sparse_results):
            key = r.chunk_id or r.text[:100]
            rrf_scores[key] += 1 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = r

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        results = []
        for key in sorted_keys[:top_k]:
            result = doc_map[key]
            result.score = rrf_scores[key]
            results.append(result)

        return results

    async def _reranked_retrieve(self, query: str, top_k: int, collection: str) -> list[RetrievalResult]:
        """Dense retrieval + cross-encoder reranking."""
        # Get more candidates for reranking
        candidates = await self._dense_retrieve(query, top_k * 3, collection)
        if not candidates or not self.embedding_engine:
            return candidates[:top_k]

        # Rerank with cross-encoder
        doc_dicts = [{"text": r.text, "score": r.score, **r.metadata} for r in candidates]
        reranked = self.embedding_engine.rerank(query, doc_dicts, top_k)

        return [
            RetrievalResult(
                text=r.text,
                score=r.reranked_score,
                metadata=r.metadata,
            )
            for r in reranked
        ]

    # ── Stats ─────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "strategy_performance": self.selector.get_stats(),
        }
