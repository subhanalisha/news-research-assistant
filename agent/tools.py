"""
Member B — MCP Tool Registry
All tools the agent can call, with typed inputs/outputs
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "../ingestion"))
from embedder import query_collection, get_parent_collection


# ── Tool implementations ──────────────────────────────────────────────────────

def search_news(query: str, k: int = 6) -> list[dict]:
    """Embed query and retrieve top-k chunks from ChromaDB."""
    chunks = query_collection(query, k=k)
    print(f"[Tool: search_news] '{query}' → {len(chunks)} chunks")
    return chunks


def filter_by_date(chunks: list[dict], days: int = 7) -> list[dict]:
    """Filter chunks to articles published within the last N days."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    filtered = []
    for chunk in chunks:
        try:
            date = datetime.fromisoformat(chunk["date"].replace("Z", "+00:00"))
            if date.replace(tzinfo=None) >= cutoff:
                filtered.append(chunk)
        except Exception:
            filtered.append(chunk)  # keep if date unparseable
    print(f"[Tool: filter_by_date] {len(chunks)} → {len(filtered)} chunks (last {days} days)")
    return filtered


def filter_by_source(chunks: list[dict], sources: list[str]) -> list[dict]:
    """Narrow results to specific news sources."""
    sources_lower = [s.lower() for s in sources]
    filtered = [c for c in chunks if c.get("source", "").lower() in sources_lower]
    print(f"[Tool: filter_by_source] {len(chunks)} → {len(filtered)} chunks")
    return filtered


def fetch_full_article(article_id: str) -> str:
    """Fetch all chunks for a given article_id and return joined text."""
    collection = get_parent_collection()
    results = collection.get(where={"article_id": article_id})
    if not results["documents"]:
        return "Article not found."
    # Sort by chunk_index and join
    pairs = sorted(zip(results["metadatas"], results["documents"]),
                   key=lambda x: x[0].get("chunk_index", 0))
    full_text = " ".join(doc for _, doc in pairs)
    print(f"[Tool: fetch_full_article] Retrieved {len(pairs)} chunks for article {article_id}")
    return full_text


def broaden_search(query: str, k: int = 12) -> list[dict]:
    """Re-run search with larger k when initial results are thin."""
    print(f"[Tool: broaden_search] Broadening search for '{query}' with k={k}")
    return query_collection(query, k=k)


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Re-rank chunks by relevance score (simple keyword overlap for now)."""
    # TODO: Replace with cross-encoder reranker for higher precision
    query_words = set(query.lower().split())
    for chunk in chunks:
        text_words = set(chunk["text"].lower().split())
        chunk["rerank_score"] = len(query_words & text_words) / max(len(query_words), 1)
    reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
    print(f"[Tool: rerank_chunks] Reranked {len(reranked)} chunks")
    return reranked


def generate_summary(query: str, chunks: list[dict]) -> str:
    """Assemble context prompt and call the LLM to produce a cited answer."""
    from dotenv import load_dotenv
    load_dotenv()

    if not chunks:
        return "No relevant articles found in the knowledge base."

    # Deduplicate — keep highest-scoring chunk per article
    seen = {}
    for chunk in chunks:
        aid = chunk.get("article_id", chunk["chunk_id"])
        if aid not in seen or chunk.get("score", 1) < seen[aid].get("score", 1):
            seen[aid] = chunk
    unique_chunks = list(seen.values())[:5]  # max 5 sources

    # Build context block
    context = ""
    for i, chunk in enumerate(unique_chunks, 1):
        context += f"[{i}] {chunk['source']}, {chunk['date'][:10]}\n{chunk['text']}\n\n"

    prompt = f"""You are a research assistant. Answer using ONLY the articles below.
Cite each source inline as [Source, Date]. If the answer is not in the articles, say so.

Articles:
{context}
Question: {query}

Answer (with inline citations):"""

    import openai
    from dotenv import load_dotenv
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        max_completion_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    print(f"[Tool: generate_summary] Answer generated ({len(answer)} chars)")
    return answer


def log_to_evals(query: str, chunks: list[dict], answer: str, scores: dict = None) -> None:
    """Persist run data to the eval harness for scoring."""
    import json
    from pathlib import Path

    log_dir = Path("../evals/results")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "chunks_retrieved": len(chunks),
        "answer": answer,
        "scores": scores or {},
        "sources": [c.get("source") for c in chunks],
    }

    log_file = log_dir / f"run_{timestamp}.json"
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2)
    print(f"[Tool: log_to_evals] Logged to {log_file}")


# ── MCP Tool Schema (for agent) ───────────────────────────────────────────────

MCP_TOOLS = [
    {
        "name": "search_news",
        "description": "Retrieve top-k relevant news chunks for a query from ChromaDB.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 6}
            },
            "required": ["query"]
        }
    },
    {
        "name": "filter_by_date",
        "description": "Filter retrieved chunks to articles published within the last N days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunks": {"type": "array"},
                "days": {"type": "integer", "default": 7}
            },
            "required": ["chunks", "days"]
        }
    },
    {
        "name": "filter_by_source",
        "description": "Narrow results to specific news sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunks": {"type": "array"},
                "sources": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["chunks", "sources"]
        }
    },
    {
        "name": "fetch_full_article",
        "description": "Fetch the full text of a specific article by article_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "article_id": {"type": "string"}
            },
            "required": ["article_id"]
        }
    },
    {
        "name": "broaden_search",
        "description": "Re-run search with larger k when initial results are thin.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 12}
            },
            "required": ["query"]
        }
    },
    {
        "name": "rerank_chunks",
        "description": "Re-rank chunks using cross-encoder for higher precision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chunks": {"type": "array"}
            },
            "required": ["query", "chunks"]
        }
    },
    {
        "name": "generate_summary",
        "description": "Assemble context prompt and call LLM to produce a cited answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chunks": {"type": "array"}
            },
            "required": ["query", "chunks"]
        }
    },
    {
        "name": "log_to_evals",
        "description": "Persist run data to the eval harness for scoring.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chunks": {"type": "array"},
                "answer": {"type": "string"},
                "scores": {"type": "object"}
            },
            "required": ["query", "chunks", "answer"]
        }
    },
]

# Tool dispatcher
TOOL_MAP = {
    "search_news": search_news,
    "filter_by_date": filter_by_date,
    "filter_by_source": filter_by_source,
    "fetch_full_article": fetch_full_article,
    "broaden_search": broaden_search,
    "rerank_chunks": rerank_chunks,
    "generate_summary": generate_summary,
    "log_to_evals": log_to_evals,
}


def execute_tool(tool_name: str, inputs: dict):
    """Dispatch a tool call by name."""
    if tool_name not in TOOL_MAP:
        return f"Unknown tool: {tool_name}"
    return TOOL_MAP[tool_name](**inputs)
