"""
Member B — MCP Tool Registry
All tools the agent can call, with typed inputs/outputs
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), "../ingestion"))
from embedder import query_collection, get_parent_collection

# ── Cross-encoder for reranking (lazy loaded on first use) ────────────────────
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("[Reranker] Loading cross-encoder model...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[Reranker] Model loaded.")
    return _cross_encoder


# ── Tool implementations ──────────────────────────────────────────────────────

def search_news(query: str, k: int = 10, **kwargs) -> list[dict]:
    """Embed query and retrieve top-k chunks from ChromaDB."""
    chunks = query_collection(query, k=k)
    print(f"[Tool: search_news] '{query}' → {len(chunks)} chunks")
    return chunks


def _parse_date(date_str: str):
    """Parse dates in both ISO and RFC-2822 (RSS) formats. Returns datetime or None."""
    if not date_str:
        return None
    # Try ISO 8601 first (NewsAPI: "2026-06-09T...")
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    # Try RFC 2822 (RSS: "Tue, 09 Jun 2026 14:30:00 GMT")
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str).replace(tzinfo=None)
    except Exception:
        pass
    return None


def filter_by_date(chunks: list[dict], days: int = 7, **kwargs) -> list[dict]:
    """Filter chunks to articles published within the last N days.
    Handles both ISO 8601 and RFC 2822 date formats.
    Falls back to original list if filtering leaves fewer than 3 chunks.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    filtered = []
    unparseable = []
    for chunk in chunks:
        parsed = _parse_date(chunk.get("date", ""))
        if parsed is None:
            unparseable.append(chunk)   # can't determine date — keep separately
        elif parsed >= cutoff:
            filtered.append(chunk)

    result = filtered + unparseable
    # Safety net: never return fewer than 3 chunks (avoids over-filtering)
    if len(result) < 3:
        print(f"[Tool: filter_by_date] {len(chunks)} → {len(result)} chunks — "
              f"too few, returning all {len(chunks)} original chunks")
        return chunks
    print(f"[Tool: filter_by_date] {len(chunks)} → {len(result)} chunks (last {days} days)")
    return result


def filter_by_source(chunks: list[dict], sources: list[str], **kwargs) -> list[dict]:
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


def broaden_search(query: str, k: int = 20, **kwargs) -> list[dict]:
    """Re-run search with larger k when initial results are thin."""
    print(f"[Tool: broaden_search] Broadening search for '{query}' with k={k}")
    return query_collection(query, k=k)


def rerank_chunks(query: str, chunks: list[dict], **kwargs) -> list[dict]:
    """Re-rank chunks using cross-encoder for higher precision."""
    if not chunks:
        return chunks
    try:
        model = get_cross_encoder()
        pairs = [(query, c.get("text", "")) for c in chunks]
        scores = model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
        print(f"[Tool: rerank_chunks] Cross-encoder reranked {len(reranked)} chunks")
        return reranked
    except Exception as e:
        print(f"[Tool: rerank_chunks] Cross-encoder failed ({e}), using keyword fallback")
        query_words = set(query.lower().split())
        for chunk in chunks:
            text_words = set(chunk.get("text", "").lower().split())
            chunk["rerank_score"] = len(query_words & text_words) / max(len(query_words), 1)
        return sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)


def rewrite_query(query: str) -> str:
    """Rewrite user query into a keyword-rich search query for better retrieval."""
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        max_completion_tokens=80,
        messages=[{
            "role": "user",
            "content": f"""Rewrite this search query to improve news article retrieval.
Make it more specific and keyword-rich. Return ONLY the rewritten query, nothing else.

Original: {query}
Rewritten:"""
        }]
    )
    rewritten = response.choices[0].message.content.strip()
    print(f"[Tool: rewrite_query] '{query}' → '{rewritten}'")
    return rewritten


def generate_summary(query: str, chunks: list[dict], **kwargs) -> str:
    """Assemble context prompt and call the LLM to produce a cited answer."""
    from dotenv import load_dotenv
    load_dotenv()

    if not chunks:
        return "No relevant articles found in the knowledge base."

    # Deduplicate by chunk_id (NOT article_id) — preserves multiple parent chunks
    # from the same article, which are different text segments with different content.
    seen_ids = set()
    unique_chunks = []
    for chunk in chunks:
        cid = chunk.get("chunk_id", str(id(chunk)))
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append(chunk)
    unique_chunks = unique_chunks[:12]  # up to 12 unique parent chunks

    # Build numbered context block
    context = ""
    for i, chunk in enumerate(unique_chunks, 1):
        source = chunk.get("source") or "Unknown"
        date   = (chunk.get("date") or "")[:10]
        title  = chunk.get("title") or ""
        text   = chunk.get("text") or ""
        context += f"[{i}] Source: {source} | Date: {date} | Title: {title}\n{text}\n\n"

    num_articles = len(unique_chunks)
    prompt = f"""Answer the question using ONLY the text in the articles below.

Rules (follow strictly):
1. Before writing any fact, verify the exact words appear in one of the articles.
2. Prefer quoting or closely paraphrasing the article text over rewording.
3. Every factual claim must be cited as [Article N — Source, Date].
4. Do NOT add any name, number, product, or event that is not written in the article text.
5. If an article is relevant, mention something from it. Skip only if truly irrelevant.
6. End with: "Summary: [one sentence citing the main source]."

ARTICLES:
{context}
QUESTION: {query}

GROUNDED ANSWER:"""


    import openai
    from dotenv import load_dotenv
    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        max_completion_tokens=1200,
        temperature=0,   # deterministic — reduces hallucination
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict news research assistant. "
                    "RULE: Only use words, names, and numbers that appear literally in the provided articles. "
                    "If a detail is not in the article text, do not include it. "
                    "When uncertain whether a detail is in the article, omit it."
                )
            },
            {"role": "user", "content": prompt},
        ]
    )
    answer = response.choices[0].message.content
    print(f"[Tool: generate_summary] Answer generated ({len(answer)} chars)")
    return answer


def log_to_evals(query: str, chunks: list[dict], answer: str = "", scores: dict = None, tool_trace: list = None, **kwargs) -> None:
    """Persist run data to the eval harness for scoring."""
    import json
    from pathlib import Path

    log_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../evals/results"))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "chunks_retrieved": chunks,
        "chunks_count": len(chunks),
        "answer": answer,
        "scores": scores or {},
        "sources": [c.get("source") for c in chunks],
        "tool_trace": tool_trace or [],
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
        "name": "rewrite_query",
        "description": "Rewrite the user query into a keyword-rich search query for better retrieval. Call this before search_news.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
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
    "rewrite_query": rewrite_query,
    "rerank_chunks": rerank_chunks,
    "generate_summary": generate_summary,
    "log_to_evals": log_to_evals,
}


def execute_tool(tool_name: str, inputs: dict):
    """Dispatch a tool call by name."""
    if tool_name not in TOOL_MAP:
        return f"Unknown tool: {tool_name}"
    return TOOL_MAP[tool_name](**inputs)
