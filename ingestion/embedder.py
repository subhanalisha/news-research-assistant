"""
Member A — Step 4 & 5: Embed chunks and store in ChromaDB
Hierarchical strategy:
  - child_chunks  collection → embedded, used for retrieval
  - parent_chunks collection → stored as text only, fetched by parent_id
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Always resolve chroma_db relative to this file, not the working directory
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

CHILD_COLLECTION  = "news_child_chunks"   # embedded — used for retrieval
PARENT_COLLECTION = "news_parent_chunks"  # text only — used for LLM context


def get_embedding_function():
    """Return the embedding function based on available API keys."""
    if OPENAI_API_KEY:
        print("[Embedder] Using OpenAI text-embedding-3-small")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
    else:
        print("[Embedder] Using local all-MiniLM-L6-v2")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )


def get_child_collection():
    """Child collection — embedded for semantic search."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = get_embedding_function()
    return client.get_or_create_collection(
        name=CHILD_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )


def get_parent_collection():
    """Parent collection — no embedding function, stored as plain text."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name=PARENT_COLLECTION)


def _safe_meta(c: dict) -> dict:
    """Ensure all metadata values are ChromaDB-safe types."""
    return {
        "article_id":  str(c.get("article_id") or ""),
        "chunk_index": int(c.get("chunk_index") or 0),
        "chunk_level": str(c.get("chunk_level") or ""),
        "parent_id":   str(c.get("parent_id") or ""),
        "title":       str(c.get("title") or ""),
        "source":      str(c.get("source") or ""),
        "date":        str(c.get("date") or ""),
        "url":         str(c.get("url") or ""),
    }


def embed_and_store(parent_chunks: list[dict], child_chunks: list[dict]):
    """
    Store parent chunks as plain text and embed child chunks for retrieval.
    """
    child_col  = get_child_collection()
    parent_col = get_parent_collection()

    # ── Store parent chunks (no embedding needed) ─────────────────────────────
    batch_size = 100
    for i in range(0, len(parent_chunks), batch_size):
        batch = parent_chunks[i:i+batch_size]
        parent_col.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[_safe_meta(c) for c in batch],
        )
    print(f"[Embedder] Stored {len(parent_chunks)} parent chunks")

    # ── Embed and store child chunks ──────────────────────────────────────────
    for i in range(0, len(child_chunks), batch_size):
        batch = child_chunks[i:i+batch_size]
        child_col.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[_safe_meta(c) for c in batch],
        )
    print(f"[Embedder] Stored {len(child_chunks)} child chunks (embedded)")


def query_collection(query: str, k: int = 10, filters: dict = None) -> list[dict]:
    """
    Hierarchical retrieval:
    1. Search child chunks for precise matches
    2. Fetch their parent chunks for richer LLM context
    Returns up to k unique parent chunks, preserving relevance score from child search.
    """
    child_col  = get_child_collection()
    parent_col = get_parent_collection()

    # Step 1 — retrieve top-k child chunks
    results = child_col.query(
        query_texts=[query],
        n_results=k,
        where=filters,
    )

    if not results["ids"][0]:
        return []

    # Step 2 — collect unique parent IDs, keeping best (lowest distance) score per parent
    pid_to_score = {}
    for j, meta in enumerate(results["metadatas"][0]):
        pid   = meta.get("parent_id", "")
        score = results["distances"][0][j]
        if pid and (pid not in pid_to_score or score < pid_to_score[pid]):
            pid_to_score[pid] = score

    parent_ids = list(pid_to_score.keys())

    # Step 3 — fetch parent chunks for LLM context
    chunks = []
    if parent_ids:
        parent_results = parent_col.get(ids=parent_ids)
        for i in range(len(parent_results["ids"])):
            pid = parent_results["ids"][i]
            chunks.append({
                "chunk_id": pid,
                "text":     parent_results["documents"][i],
                "score":    pid_to_score.get(pid, 1.0),  # correct: child's score for this parent
                **parent_results["metadatas"][i],
            })
        # Sort by relevance (lower cosine distance = more relevant)
        chunks.sort(key=lambda x: x["score"])
    else:
        # Fallback: return child chunks directly
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text":     results["documents"][0][i],
                "score":    results["distances"][0][i],
                **results["metadatas"][0][i],
            })

    return chunks


if __name__ == "__main__":
    # Quick test
    sample_parents = [{
        "chunk_id": "parent-001", "article_id": "art-001", "chunk_index": 0,
        "chunk_level": "parent", "parent_id": None,
        "text": "OpenAI released GPT-5 with improved reasoning. The model scores higher on all benchmarks.",
        "title": "GPT-5 Released", "source": "TechCrunch", "date": "2026-06-01", "url": "https://example.com",
    }]
    sample_children = [{
        "chunk_id": "child-001", "article_id": "art-001", "chunk_index": 0,
        "chunk_level": "child", "parent_id": "parent-001",
        "text": "OpenAI released GPT-5 with improved reasoning.",
        "title": "GPT-5 Released", "source": "TechCrunch", "date": "2026-06-01", "url": "https://example.com",
    }]
    embed_and_store(sample_parents, sample_children)
    results = query_collection("OpenAI new model", k=1)
    print(f"Query result: {results[0]['text'][:80] if results else 'No results'}")
