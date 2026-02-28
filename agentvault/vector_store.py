"""
Shield Command — Vector Store

ChromaDB-backed vector database with Shield-specific features.
Stores document embeddings for semantic search, RAG retrieval,
and memory augmentation.

Runs embedded (in-process) — no server needed.
Falls back to a pure-Python cosine similarity store if ChromaDB
is not installed, ensuring the system always works.

Integration points:
    - EmbeddingEngine: provides vectors
    - RetrievalCortex: primary search backend
    - SemanticCache: similarity-based cache lookup
    - CognitiveMemory: semantic search over episodes
"""

import json
import time
import uuid
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("shield.vector_store")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class VectorDocument:
    """A document stored in the vector store."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    text: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    collection: str = "default"
    created_at: float = field(default_factory=time.time)


@dataclass
class SimilarityResult:
    """A search result with similarity score."""
    id: str = ""
    text: str = ""
    score: float = 0.0       # 0-1, higher = more similar
    distance: float = 0.0    # lower = more similar
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ChromaDB Backend
# ---------------------------------------------------------------------------

class ChromaBackend:
    """ChromaDB-based vector store backend."""

    def __init__(self, persist_dir: str = "shield_vectors"):
        self.persist_dir = persist_dir
        self._client = None
        self._collections: dict = {}
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import chromadb
            self._available = True
        except ImportError:
            self._available = False
            logger.info("chromadb not installed — using fallback vector store")
        return self._available

    def _get_client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_dir,
                anonymized_telemetry=False,
            ))
        return self._client

    def _get_collection(self, name: str):
        if name not in self._collections:
            client = self._get_client()
            self._collections[name] = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    async def add(self, doc: VectorDocument):
        collection = self._get_collection(doc.collection)
        collection.add(
            ids=[doc.id],
            embeddings=[doc.embedding],
            documents=[doc.text],
            metadatas=[doc.metadata] if doc.metadata else None,
        )

    async def add_batch(self, docs: list[VectorDocument]):
        if not docs:
            return
        # Group by collection
        by_collection: dict[str, list] = {}
        for doc in docs:
            by_collection.setdefault(doc.collection, []).append(doc)
        
        for coll_name, coll_docs in by_collection.items():
            collection = self._get_collection(coll_name)
            collection.add(
                ids=[d.id for d in coll_docs],
                embeddings=[d.embedding for d in coll_docs],
                documents=[d.text for d in coll_docs],
                metadatas=[d.metadata for d in coll_docs if d.metadata] or None,
            )

    async def search(
        self,
        query_embedding: list[float],
        collection: str = "default",
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[SimilarityResult]:
        coll = self._get_collection(collection)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = coll.query(**kwargs)

        output = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0
                score = max(0.0, 1.0 - distance)  # cosine distance → similarity
                output.append(SimilarityResult(
                    id=doc_id,
                    text=results["documents"][0][i] if results.get("documents") else "",
                    score=score,
                    distance=distance,
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                ))
        return output

    async def delete(self, doc_id: str, collection: str = "default"):
        coll = self._get_collection(collection)
        coll.delete(ids=[doc_id])

    async def count(self, collection: str = "default") -> int:
        coll = self._get_collection(collection)
        return coll.count()


# ---------------------------------------------------------------------------
# SQLite Fallback Backend (no external deps)
# ---------------------------------------------------------------------------

class SQLiteFallbackBackend:
    """
    Pure-Python vector store using SQLite for persistence
    and cosine similarity for search. Zero dependencies.
    """

    def __init__(self, db_path: str = "shield_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    embedding TEXT,
                    metadata TEXT DEFAULT '{}',
                    collection TEXT DEFAULT 'default',
                    created_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_coll ON vectors(collection)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    async def add(self, doc: VectorDocument):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vectors (id, text, embedding, metadata, collection, created_at) VALUES (?,?,?,?,?,?)",
                (doc.id, doc.text, json.dumps(doc.embedding), json.dumps(doc.metadata),
                 doc.collection, doc.created_at),
            )

    async def add_batch(self, docs: list[VectorDocument]):
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (id, text, embedding, metadata, collection, created_at) VALUES (?,?,?,?,?,?)",
                [(d.id, d.text, json.dumps(d.embedding), json.dumps(d.metadata),
                  d.collection, d.created_at) for d in docs],
            )

    async def search(
        self,
        query_embedding: list[float],
        collection: str = "default",
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[SimilarityResult]:
        from .embedding_engine import cosine_similarity

        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, text, embedding, metadata FROM vectors WHERE collection = ?",
                (collection,),
            ).fetchall()

        scored = []
        for row in rows:
            doc_embedding = json.loads(row[2])
            if len(doc_embedding) != len(query_embedding):
                continue
            score = cosine_similarity(query_embedding, doc_embedding)
            metadata = json.loads(row[3]) if row[3] else {}

            # Apply metadata filter
            if where:
                match = all(metadata.get(k) == v for k, v in where.items())
                if not match:
                    continue

            scored.append(SimilarityResult(
                id=row[0],
                text=row[1],
                score=score,
                distance=1.0 - score,
                metadata=metadata,
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    async def delete(self, doc_id: str, collection: str = "default"):
        with self._conn() as conn:
            conn.execute("DELETE FROM vectors WHERE id = ? AND collection = ?", (doc_id, collection))

    async def count(self, collection: str = "default") -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection = ?", (collection,)
            ).fetchone()[0]


# ---------------------------------------------------------------------------
# Unified Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Unified vector store with ChromaDB → SQLite fallback.
    
    Usage:
        from agentvault.embedding_engine import EmbeddingEngine
        from agentvault.vector_store import VectorStore, VectorDocument
        
        engine = EmbeddingEngine()
        store = VectorStore()
        
        # Add a document
        result = await engine.embed("Shield is an AI security platform")
        doc = VectorDocument(text="Shield is an AI security platform", embedding=result.vector)
        await store.add(doc)
        
        # Search
        query_vec = await engine.embed("What is Shield?")
        results = await store.search(query_vec.vector, top_k=5)
        for r in results:
            print(f"  [{r.score:.3f}] {r.text}")
    """

    def __init__(
        self,
        persist_dir: str = "shield_vectors",
        db_path: str = "shield_memory.db",
    ):
        self._chroma = ChromaBackend(persist_dir=persist_dir)
        self._fallback = SQLiteFallbackBackend(db_path=db_path)
        self._backend = None

        # Stats
        self.total_adds = 0
        self.total_searches = 0

    @property
    def backend(self):
        if self._backend is None:
            if self._chroma.is_available():
                self._backend = self._chroma
                logger.info("VectorStore: using ChromaDB backend")
            else:
                self._backend = self._fallback
                logger.info("VectorStore: using SQLite fallback backend")
        return self._backend

    @property
    def backend_name(self) -> str:
        return "chromadb" if isinstance(self.backend, ChromaBackend) else "sqlite_fallback"

    async def add(self, doc: VectorDocument):
        """Add a single document."""
        self.total_adds += 1
        await self.backend.add(doc)

    async def add_batch(self, docs: list[VectorDocument]):
        """Add a batch of documents."""
        self.total_adds += len(docs)
        await self.backend.add_batch(docs)

    async def search(
        self,
        query_embedding: list[float],
        collection: str = "default",
        top_k: int = 10,
        where: Optional[dict] = None,
        min_score: float = 0.0,
    ) -> list[SimilarityResult]:
        """Search for similar documents."""
        self.total_searches += 1
        results = await self.backend.search(query_embedding, collection, top_k, where)
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        return results

    async def delete(self, doc_id: str, collection: str = "default"):
        """Delete a document."""
        await self.backend.delete(doc_id, collection)

    async def count(self, collection: str = "default") -> int:
        """Count documents in a collection."""
        return await self.backend.count(collection)

    async def get_stats(self) -> dict:
        default_count = await self.count("default")
        return {
            "backend": self.backend_name,
            "total_adds": self.total_adds,
            "total_searches": self.total_searches,
            "default_collection_size": default_count,
        }
