"""
Member A — Step 3: Hierarchical Chunking Strategy
- Parent chunks (~1500 tokens): full paragraph context, sent to LLM
- Child chunks (~150 tokens): precise sentences, used for retrieval
Child chunks store a reference to their parent so the LLM always
receives the richer parent context.
"""

import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Parent: large context block sent to LLM
PARENT_CHUNK_SIZE    = 1500   # tokens
PARENT_CHUNK_OVERLAP = 100    # tokens

# Child: small precise unit used for retrieval
CHILD_CHUNK_SIZE    = 150     # tokens
CHILD_CHUNK_OVERLAP = 20      # tokens

CHARS_PER_TOKEN = 4  # rough approximation


def chunk_article(article: dict) -> tuple[list[dict], list[dict]]:
    """
    Split one article into parent and child chunks.
    Returns (parent_chunks, child_chunks).
    Child chunks carry a parent_id so context can be fetched at query time.
    """
    article_id = article.get("article_id", str(uuid.uuid4()))
    base_meta = {
        "article_id": article_id,
        "title":  article.get("title", ""),
        "source": article.get("source", ""),
        "date":   article.get("date", ""),
        "url":    article.get("url", ""),
    }

    # ── Parent splitter ───────────────────────────────────────────────────────
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE * CHARS_PER_TOKEN,
        chunk_overlap=PARENT_CHUNK_OVERLAP * CHARS_PER_TOKEN,
        separators=["\n\n", "\n", ". ", " "],
    )

    # ── Child splitter ────────────────────────────────────────────────────────
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE * CHARS_PER_TOKEN,
        chunk_overlap=CHILD_CHUNK_OVERLAP * CHARS_PER_TOKEN,
        separators=[". ", "! ", "? ", "\n", " "],
    )

    parent_texts = parent_splitter.split_text(article["content"])
    parent_chunks = []
    child_chunks  = []

    for p_idx, p_text in enumerate(parent_texts):
        parent_id = str(uuid.uuid4())

        # Store parent chunk
        parent_chunks.append({
            "chunk_id":    parent_id,
            "chunk_index": p_idx,
            "chunk_level": "parent",
            "parent_id":   None,        # parents have no parent
            "text":        p_text,
            **base_meta,
        })

        # Split parent into children
        child_texts = child_splitter.split_text(p_text)
        for c_idx, c_text in enumerate(child_texts):
            child_chunks.append({
                "chunk_id":    str(uuid.uuid4()),
                "chunk_index": c_idx,
                "chunk_level": "child",
                "parent_id":   parent_id,  # ← link back to parent
                "text":        c_text,
                **base_meta,
            })

    return parent_chunks, child_chunks


def chunk_all_articles(articles: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Chunk all articles hierarchically.
    Returns (all_parent_chunks, all_child_chunks).
    """
    for article in articles:
        if "article_id" not in article:
            article["article_id"] = str(uuid.uuid4())

    all_parents = []
    all_children = []

    for article in articles:
        parents, children = chunk_article(article)
        all_parents.extend(parents)
        all_children.extend(children)

    print(f"[Chunker] {len(articles)} articles → "
          f"{len(all_parents)} parent chunks, {len(all_children)} child chunks")
    return all_parents, all_children


if __name__ == "__main__":
    sample = [{
        "title": "OpenAI releases GPT-5",
        "source": "TechCrunch",
        "date": "2026-06-01",
        "url": "https://example.com",
        "content": "OpenAI has released GPT-5. " * 200,
    }]
    parents, children = chunk_all_articles(sample)
    print(f"Parents: {len(parents)}, Children: {len(children)}")
    print(f"First child text : {children[0]['text'][:80]}")
    print(f"Child parent_id  : {children[0]['parent_id']}")
    print(f"Matching parent  : {parents[0]['chunk_id'] == children[0]['parent_id']}")
