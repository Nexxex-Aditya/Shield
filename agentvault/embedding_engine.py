"""
Shield Command — Embedding Engine

Local-first embedding pipeline with cloud fallback.
Converts text into dense vector representations for semantic search,
memory retrieval, and caching.

Components:
    - LocalEmbedder: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
    - CloudEmbedder: OpenAI text-embedding-3-small fallback
    - CrossEncoderReranker: re-scores retrieved documents for precision
    - EmbeddingEngine: unified interface with auto-detection

Integration points:
    - VectorStore: provides embeddings for storage
    - SemanticCache: embeds queries for near-match lookup
    - RetrievalCortex: embeds queries + reranks results
    - CognitiveMemory: upgrades EpisodicMemory search to semantic
"""

import logging
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache

logger = logging.getLogger("shield.embedding")


# ---------------------------------------------------------------------------
# Embedding Result
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    vector: list[float]
    model: str = ""
    dimensions: int = 0
    tokens_used: int = 0
    latency_ms: float = 0.0
    source: str = "local"  # "local" or "cloud"


@dataclass
class RerankedResult:
    """A search result after reranking."""
    text: str
    original_score: float
    reranked_score: float
    metadata: dict = field(default_factory=dict)
    rank: int = 0


# ---------------------------------------------------------------------------
# Local Embedder (sentence-transformers)
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """
    Generates embeddings using sentence-transformers locally.
    Zero API cost, runs on CPU.
    
    Model: all-MiniLM-L6-v2 (384 dimensions, ~80MB)
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._available = None

    def is_available(self) -> bool:
        """Check if sentence-transformers is installed and model is loadable."""
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer
            self._available = True
        except ImportError:
            self._available = False
            logger.info("sentence-transformers not installed — will use cloud embeddings")
        return self._available

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded local embedding model: {self.model_name}")

    def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text string."""
        self._load_model()
        start = time.time()
        vector = self._model.encode(text, normalize_embeddings=True).tolist()
        return EmbeddingResult(
            vector=vector,
            model=self.model_name,
            dimensions=len(vector),
            latency_ms=(time.time() - start) * 1000,
            source="local",
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts efficiently."""
        self._load_model()
        start = time.time()
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        elapsed = (time.time() - start) * 1000
        per_item = elapsed / max(len(texts), 1)
        return [
            EmbeddingResult(
                vector=v.tolist(),
                model=self.model_name,
                dimensions=len(v),
                latency_ms=per_item,
                source="local",
            )
            for v in vectors
        ]


# ---------------------------------------------------------------------------
# Cloud Embedder (OpenAI)
# ---------------------------------------------------------------------------

class CloudEmbedder:
    """
    Generates embeddings via OpenAI API.
    Fallback when local models are unavailable.
    
    Model: text-embedding-3-small (1536 dimensions)
    """

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model
        self._client = None

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai
            return True
        except ImportError:
            return False

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text via OpenAI API."""
        start = time.time()
        client = self._get_client()
        response = client.embeddings.create(input=text, model=self.model)
        vector = response.data[0].embedding
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        return EmbeddingResult(
            vector=vector,
            model=self.model,
            dimensions=len(vector),
            tokens_used=tokens,
            latency_ms=(time.time() - start) * 1000,
            source="cloud",
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts via OpenAI API."""
        start = time.time()
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self.model)
        elapsed = (time.time() - start) * 1000
        per_item = elapsed / max(len(texts), 1)
        results = []
        for item in response.data:
            results.append(EmbeddingResult(
                vector=item.embedding,
                model=self.model,
                dimensions=len(item.embedding),
                tokens_used=0,
                latency_ms=per_item,
                source="cloud",
            ))
        return results


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Re-scores query-document pairs using a cross-encoder for high-precision retrieval.
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import CrossEncoder
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder reranker: {self.model_name}")

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        text_key: str = "text",
    ) -> list[RerankedResult]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: search query
            documents: list of dicts with at least a text_key field
            top_k: return top K results
            text_key: key to extract text from document dicts
        """
        if not documents:
            return []

        if not self.is_available():
            # Fallback: return documents in original order with placeholder scores
            return [
                RerankedResult(
                    text=doc.get(text_key, ""),
                    original_score=doc.get("score", 0),
                    reranked_score=doc.get("score", 0),
                    metadata={k: v for k, v in doc.items() if k != text_key},
                    rank=i,
                )
                for i, doc in enumerate(documents[:top_k])
            ]

        self._load_model()

        pairs = [(query, doc.get(text_key, "")) for doc in documents]
        scores = self._model.predict(pairs)

        scored = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            scored.append(RerankedResult(
                text=doc.get(text_key, ""),
                original_score=doc.get("score", 0),
                reranked_score=float(score),
                metadata={k: v for k, v in doc.items() if k != text_key},
            ))

        scored.sort(key=lambda x: x.reranked_score, reverse=True)
        for i, r in enumerate(scored):
            r.rank = i

        return scored[:top_k]


# ---------------------------------------------------------------------------
# Cosine Similarity (pure Python fallback)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Unified Embedding Engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Unified embedding interface with auto-detection.
    
    Priority: Local (free) → Cloud (paid fallback)
    
    Usage:
        engine = EmbeddingEngine(openai_api_key="sk-...")
        
        # Single embedding
        result = await engine.embed("What is Shield?")
        print(result.vector)  # [0.012, -0.034, ...]
        
        # Batch embedding
        results = await engine.embed_batch(["doc1", "doc2", "doc3"])
        
        # Reranking
        ranked = engine.rerank("search query", [{"text": "doc1"}, {"text": "doc2"}])
    """

    def __init__(
        self,
        openai_api_key: str = "",
        local_model: str = LocalEmbedder.DEFAULT_MODEL,
        cloud_model: str = CloudEmbedder.DEFAULT_MODEL,
    ):
        self.local = LocalEmbedder(local_model)
        self.cloud = CloudEmbedder(api_key=openai_api_key, model=cloud_model)
        self.reranker = CrossEncoderReranker()
        self._embedding_cache: dict[str, list[float]] = {}
        self._cache_max = 10000

        # Stats
        self.total_embeds = 0
        self.local_embeds = 0
        self.cloud_embeds = 0
        self.cache_hits = 0

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions for the active model."""
        if self.local.is_available():
            return 384  # MiniLM
        return 1536  # OpenAI

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text. Uses cache → local → cloud."""
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            self.cache_hits += 1
            return EmbeddingResult(
                vector=self._embedding_cache[cache_key],
                model="cache",
                dimensions=len(self._embedding_cache[cache_key]),
                source="cache",
            )

        self.total_embeds += 1

        # Try local first
        if self.local.is_available():
            result = self.local.embed(text)
            self.local_embeds += 1
        elif self.cloud.is_available():
            result = await self.cloud.embed(text)
            self.cloud_embeds += 1
        else:
            raise RuntimeError(
                "No embedding backend available. Install sentence-transformers or provide openai_api_key."
            )

        # Cache the result
        if len(self._embedding_cache) < self._cache_max:
            self._embedding_cache[cache_key] = result.vector

        return result

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a batch of texts."""
        if not texts:
            return []

        self.total_embeds += len(texts)

        if self.local.is_available():
            results = self.local.embed_batch(texts)
            self.local_embeds += len(texts)
        elif self.cloud.is_available():
            results = await self.cloud.embed_batch(texts)
            self.cloud_embeds += len(texts)
        else:
            raise RuntimeError("No embedding backend available.")

        # Cache results
        for text, result in zip(texts, results):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if len(self._embedding_cache) < self._cache_max:
                self._embedding_cache[cache_key] = result.vector

        return results

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Rerank documents using cross-encoder."""
        return self.reranker.rerank(query, documents, top_k)

    async def get_stats(self) -> dict:
        return {
            "total_embeds": self.total_embeds,
            "local_embeds": self.local_embeds,
            "cloud_embeds": self.cloud_embeds,
            "cache_hits": self.cache_hits,
            "cache_size": len(self._embedding_cache),
            "local_available": self.local.is_available(),
            "cloud_available": self.cloud.is_available(),
            "reranker_available": self.reranker.is_available(),
            "active_model": self.local.model_name if self.local.is_available() else self.cloud.model,
            "dimensions": self.dimensions,
        }
