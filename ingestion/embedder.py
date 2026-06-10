"""
Member A — Step 4 & 5: Embed chunks and store in ChromaDB
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Always resolve chroma_db relative to this file, not the working directory
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "news_articles"


def get_collection():
    """Initialize ChromaDB client and return the collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Use OpenAI embeddings if key available, else free local model
    if OPENAI_API_KEY:
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        print("[Embedder] Using OpenAI text-embedding-3-small")
    else:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        print("[Embedder] Using local all-MiniLM-L6-v2")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def embed_and_store(chunks: list[dict]):
    """Embed chunks and upsert into ChromaDB."""
    collection = get_collection()

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{
        "article_id": str(c.get("article_id") or ""),
        "chunk_index": int(c.get("chunk_index") or 0),
        "title": str(c.get("title") or ""),
        "source": str(c.get("source") or ""),
        "date": str(c.get("date") or ""),
        "url": str(c.get("url") or ""),
    } for c in chunks]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )

    print(f"[Embedder] Stored {len(chunks)} chunks in ChromaDB")


def query_collection(query: str, k: int = 6, filters: dict = None) -> list[dict]:
    """Retrieve top-k chunks for a query."""
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=filters,
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "score": results["distances"][0][i],
            **results["metadatas"][0][i],
        })
    return chunks


if __name__ == "__main__":
    # Quick test
    sample_chunks = [{
        "chunk_id": "test-001",
        "article_id": "art-001",
        "chunk_index": 0,
        "text": "OpenAI released GPT-5 with improved reasoning capabilities.",
        "title": "GPT-5 Released",
        "source": "TechCrunch",
        "date": "2026-06-01",
        "url": "https://example.com",
    }]
    embed_and_store(sample_chunks)
    results = query_collection("OpenAI new model", k=1)
    print(f"Query result: {results}")
