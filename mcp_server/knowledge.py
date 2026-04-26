"""
Knowledge Module — ChromaDB-powered RAG for Sam.

Gives Sam a brain loaded with 139+ trading book summaries,
the Options Field Manual, and the Agentic Trader's Playbook.

Uses Gemini embeddings API (free tier: 1,500 req/min).
ChromaDB stores to data/chromadb/ (project-local, persists across restarts).
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import chromadb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMADB_PATH = PROJECT_ROOT / "data" / "chromadb"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_BATCH_SIZE = 20

# ---------------------------------------------------------------------------
# Lazy-loaded clients
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.PersistentClient] = None
_genai_client = None


def _get_chroma() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        logger.info("ChromaDB initialized at %s", CHROMADB_PATH)
    return _chroma_client


def _get_genai():
    global _genai_client
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def _embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed texts using Gemini embedding API. Handles batching."""
    from google.genai import types

    client = _get_genai()
    all_embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768,
                ),
            )
            all_embeddings.extend([e.values for e in result.embeddings])
        except Exception as e:
            logger.warning("Embedding error (batch %d): %s", i, e)
            all_embeddings.extend([[0.0] * 768] * len(batch))

    return all_embeddings


def _embed_query(text: str) -> list[float]:
    """Embed a single query text."""
    results = _embed_texts([text], task_type="RETRIEVAL_QUERY")
    return results[0] if results else [0.0] * 768


# ---------------------------------------------------------------------------
# Knowledge Base Collection
# ---------------------------------------------------------------------------

def _get_knowledge_collection() -> chromadb.Collection:
    return _get_chroma().get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )


def ingest_knowledge(
    text: str,
    source: str,
    doc_type: str = "book_summary",
    section: str = "",
) -> None:
    """Ingest a chunk of knowledge into the knowledge base."""
    try:
        collection = _get_knowledge_collection()
        chunk_id = hashlib.md5(f"{source}:{section}:{text[:100]}".encode()).hexdigest()
        embedding = _embed_texts([text])[0]

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "source": source,
                "section": section,
                "doc_type": doc_type,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }],
        )
    except Exception as e:
        logger.warning("Knowledge ingest error for %s: %s", source, e)


# ---------------------------------------------------------------------------
# Search — LLM Tool
# ---------------------------------------------------------------------------

async def search_knowledge(query: str, top_k: int = 5) -> list[dict]:
    """
    Search Sam's knowledge base — 139 trading books, options manual, and methodology.

    Args:
        query: What to search for (e.g., "cash secured puts", "EMA stack", "VoPR").
        top_k: Number of results. Default: 5.

    Returns a list of relevant knowledge chunks with source attribution.
    """
    try:
        collection = _get_knowledge_collection()
        if collection.count() == 0:
            return [{"info": "Knowledge base is empty — run seed_knowledge.py first"}]

        query_embedding = _embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                score = 1 - results["distances"][0][i]  # cosine distance → similarity
                if score < 0.25:  # skip low-relevance noise
                    continue
                meta = results["metadatas"][0][i]
                output.append({
                    "text": results["documents"][0][i],
                    "source": meta.get("source", "?"),
                    "section": meta.get("section", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "relevance": round(score, 3),
                })

        return output if output else [{"info": "No relevant knowledge found for that query."}]
    except Exception as e:
        logger.warning("Knowledge search error: %s", e)
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# RAG Context Builder — auto-injection into prompts
# ---------------------------------------------------------------------------

def get_rag_context(query: str, max_chars: int = 4000) -> tuple[str, list[dict]]:
    """
    Build RAG context string for prompt injection.

    Returns (context_string, sources_list) where sources_list contains
    the source/section pairs for Glass Box transparency display.
    """
    try:
        collection = _get_knowledge_collection()
        if collection.count() == 0:
            return "", []

        query_embedding = _embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(5, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        sections = []
        sources = []
        total_chars = 0

        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                score = 1 - results["distances"][0][i]
                if score < 0.35:  # higher threshold for auto-injection
                    continue

                meta = results["metadatas"][0][i]
                text = results["documents"][0][i]

                # Trim if needed but keep it thorough
                if total_chars + len(text) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining < 200:
                        break
                    text = text[:remaining] + "..."

                source = meta.get("source", "Unknown")
                section = meta.get("section", "")
                header = f"[{source}" + (f" — {section}]" if section else "]")
                sections.append(f"{header}\n{text}")
                sources.append({
                    "source": source,
                    "section": section,
                    "relevance": round(score, 3),
                })
                total_chars += len(text) + len(header)

        if not sections:
            return "", []

        context = "## Relevant Knowledge from Sam's Library:\n\n" + "\n\n---\n\n".join(sections)
        return context, sources
    except Exception as e:
        logger.warning("RAG context error: %s", e)
        return "", []


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_knowledge_stats() -> dict:
    """Get knowledge base statistics."""
    try:
        kb = _get_knowledge_collection()
        return {
            "total_chunks": kb.count(),
            "chromadb_path": str(CHROMADB_PATH),
        }
    except Exception as e:
        return {"error": str(e)}
