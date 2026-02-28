"""
Shield Command — Document Processor

Ingests documents (text, markdown, code, JSON) into the vector store
for RAG retrieval. Handles chunking, embedding, and storage.

Components:
    - RecursiveChunker: splits documents respecting semantic boundaries
    - DocumentIngester: orchestrates chunk → embed → store pipeline
    - DocumentProcessor: unified interface

Integration points:
    - EmbeddingEngine: generates embeddings for chunks
    - VectorStore: stores embedded chunks
    - RetrievalCortex: provides the documents that RAG retrieves
"""

import re
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("shield.document_processor")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """Configuration for document chunking."""
    chunk_size: int = 512          # Target tokens per chunk
    chunk_overlap: int = 64        # Overlap between chunks (tokens)
    respect_boundaries: bool = True  # Respect paragraph/heading boundaries
    min_chunk_size: int = 50       # Minimum chunk size (skip tiny fragments)


@dataclass
class DocumentChunk:
    """A chunk of a document ready for embedding."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    text: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    source_id: str = ""        # Original document identifier
    source_name: str = ""      # Filename, URL, etc.
    doc_type: str = "text"     # text, markdown, code, json
    char_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    source_name: str = ""
    total_chunks: int = 0
    total_characters: int = 0
    embedding_model: str = ""
    duration_ms: float = 0.0
    chunk_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Recursive Chunker
# ---------------------------------------------------------------------------

class RecursiveChunker:
    """
    Splits documents into chunks respecting semantic boundaries.
    
    Priority of split points (high → low):
        1. Headings (# ## ### in markdown)
        2. Double newlines (paragraphs)
        3. Single newlines
        4. Sentence boundaries (. ! ?)
        5. Word boundaries (spaces)
        6. Character boundary (last resort)
    """

    SEPARATORS = [
        "\n## ",     # H2 heading
        "\n### ",    # H3 heading
        "\n\n",      # Paragraph
        "\n",        # Line
        ". ",        # Sentence
        "! ",        # Sentence
        "? ",        # Sentence
        "; ",        # Clause
        ", ",        # Phrase
        " ",         # Word
    ]

    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()

    def chunk(self, text: str, doc_type: str = "text") -> list[str]:
        """Split text into chunks."""
        if not text or not text.strip():
            return []

        # Estimate tokens as ~4 chars per token
        target_chars = self.config.chunk_size * 4
        overlap_chars = self.config.chunk_overlap * 4
        min_chars = self.config.min_chunk_size * 4

        chunks = self._recursive_split(text, target_chars, overlap_chars)

        # Filter out tiny chunks
        chunks = [c.strip() for c in chunks if len(c.strip()) >= min_chars]

        return chunks

    def _recursive_split(self, text: str, target: int, overlap: int) -> list[str]:
        """Recursively split text using hierarchical separators."""
        if len(text) <= target:
            return [text]

        # Find the best separator that produces reasonable chunks
        for sep in self.SEPARATORS:
            if sep in text:
                parts = text.split(sep)
                chunks = self._merge_parts(parts, sep, target, overlap)
                if len(chunks) > 1:
                    return chunks

        # Last resort: character split
        return self._char_split(text, target, overlap)

    def _merge_parts(self, parts: list[str], sep: str, target: int, overlap: int) -> list[str]:
        """Merge small parts into chunks that approach the target size."""
        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) > target and current:
                chunks.append(current)
                # Add overlap from end of previous chunk
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + sep + part
                else:
                    current = part
            else:
                current = candidate

        if current:
            chunks.append(current)

        return chunks

    def _char_split(self, text: str, target: int, overlap: int) -> list[str]:
        """Split by character count with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + target, len(text))
            chunks.append(text[start:end])
            start = end - overlap if overlap > 0 else end
        return chunks


# ---------------------------------------------------------------------------
# Document Type Detector
# ---------------------------------------------------------------------------

def detect_doc_type(text: str, filename: str = "") -> str:
    """Detect document type from content or filename."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    ext_map = {
        "md": "markdown", "markdown": "markdown",
        "py": "code", "js": "code", "ts": "code", "java": "code",
        "go": "code", "rs": "code", "cpp": "code", "c": "code",
        "json": "json", "yaml": "json", "yml": "json", "toml": "json",
        "html": "markup", "xml": "markup", "svg": "markup",
        "txt": "text", "csv": "text", "log": "text",
    }

    if ext in ext_map:
        return ext_map[ext]

    # Content-based detection
    if text.startswith("{") or text.startswith("["):
        return "json"
    if re.match(r"^#+ ", text, re.MULTILINE):
        return "markdown"
    if any(kw in text[:500] for kw in ["def ", "class ", "import ", "function ", "const "]):
        return "code"

    return "text"


# ---------------------------------------------------------------------------
# Document Processor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    Orchestrates document ingestion: chunk → embed → store.
    
    Usage:
        processor = DocumentProcessor(
            embedding_engine=engine,
            vector_store=store,
        )
        
        result = await processor.ingest(
            text="...",
            source_name="report.md",
            collection="knowledge",
        )
        print(f"Ingested {result.total_chunks} chunks")
    """

    def __init__(
        self,
        embedding_engine=None,
        vector_store=None,
        chunk_config: ChunkConfig = None,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.chunker = RecursiveChunker(chunk_config or ChunkConfig())

        # Stats
        self.total_documents = 0
        self.total_chunks = 0
        self.total_characters = 0

    async def ingest(
        self,
        text: str,
        source_name: str = "unnamed",
        collection: str = "documents",
        metadata: dict = None,
    ) -> IngestionResult:
        """
        Ingest a document: chunk → embed → store.
        """
        start_time = time.time()
        source_id = str(uuid.uuid4())[:10]
        doc_type = detect_doc_type(text, source_name)

        # Step 1: Chunk
        raw_chunks = self.chunker.chunk(text, doc_type)

        if not raw_chunks:
            return IngestionResult(source_name=source_name)

        # Build DocumentChunk objects
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_meta = {
                "source_name": source_name,
                "source_id": source_id,
                "doc_type": doc_type,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
            }
            if metadata:
                chunk_meta.update(metadata)

            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_index=i,
                total_chunks=len(raw_chunks),
                source_id=source_id,
                source_name=source_name,
                doc_type=doc_type,
                char_count=len(chunk_text),
                metadata=chunk_meta,
            ))

        # Step 2: Embed
        if not self.embedding_engine:
            raise RuntimeError("EmbeddingEngine not configured")

        embed_results = await self.embedding_engine.embed_batch(
            [c.text for c in chunks]
        )

        # Step 3: Store
        if not self.vector_store:
            raise RuntimeError("VectorStore not configured")

        from .vector_store import VectorDocument
        vector_docs = []
        for chunk, embed in zip(chunks, embed_results):
            vector_docs.append(VectorDocument(
                id=chunk.id,
                text=chunk.text,
                embedding=embed.vector,
                metadata=chunk.metadata,
                collection=collection,
            ))

        await self.vector_store.add_batch(vector_docs)

        # Update stats
        self.total_documents += 1
        self.total_chunks += len(chunks)
        self.total_characters += sum(c.char_count for c in chunks)

        result = IngestionResult(
            source_name=source_name,
            total_chunks=len(chunks),
            total_characters=sum(c.char_count for c in chunks),
            embedding_model=embed_results[0].model if embed_results else "",
            duration_ms=(time.time() - start_time) * 1000,
            chunk_ids=[c.id for c in chunks],
        )

        logger.info(
            f"Ingested '{source_name}': {result.total_chunks} chunks, "
            f"{result.total_characters} chars, {result.duration_ms:.0f}ms"
        )
        return result

    async def ingest_batch(
        self,
        documents: list[dict],
        collection: str = "documents",
    ) -> list[IngestionResult]:
        """
        Ingest multiple documents. Each dict needs: text, source_name.
        """
        results = []
        for doc in documents:
            result = await self.ingest(
                text=doc.get("text", ""),
                source_name=doc.get("source_name", "unnamed"),
                collection=collection,
                metadata=doc.get("metadata"),
            )
            results.append(result)
        return results

    async def get_stats(self) -> dict:
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_characters": self.total_characters,
        }
