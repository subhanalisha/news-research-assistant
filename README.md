# News Research Assistant
**Capstone Project 06** — RAG · MCP · Agents · Evals · 7-Day Sprint · 3 Members

An agentic news research assistant that continuously ingests articles from multiple sources,
retrieves the most relevant content using semantic search, and generates cited, grounded
multi-source summaries — with automatic quality evaluation built in.

---

## What It Does

You ask a research question. The agent:
1. **Rewrites** your query into retrieval-optimised keywords
2. **Searches** a local ChromaDB vector store (hierarchical parent-child chunks)
3. **Broadens** the search automatically if initial results are thin
4. **Filters** to recent articles when the query is time-sensitive
5. **Reranks** candidates with a cross-encoder for precision ordering
6. **Generates** a cited answer using *only* retrieved text (temperature=0, no hallucination)
7. **Logs** the full tool trace for automatic evaluation

Every answer includes inline citations: `[Article N — Source, Date]`.

---

## Architecture Overview

```
News Sources (RSS + NewsAPI)
        │
        ▼
Ingestion Pipeline  ──────────────────────────────────────────────────────
  fetcher.py    → RSS feeds (TechCrunch, Wired, Verge, Ars, VentureBeat)
                  + NewsAPI (OpenAI, AI, startups)
  chunker.py    → Hierarchical chunking:
                    Parent chunks (6000 chars) — rich LLM context
                    Child chunks  (600 chars)  — precise vector search
  embedder.py   → text-embedding-3-small (1536-dim) → ChromaDB
        │
        ▼
ChromaDB (local)
  news_child_chunks  ← embedded, searchable
  news_parent_chunks ← text only, fetched after child match
        │
        ▼
Agent Reasoning Loop (gpt-5.4-mini · MCP tools)
  rewrite_query  →  search_news (k=10)  →  [broaden_search (k=20)]
                 →  [filter_by_date (days=7)]  →  rerank_chunks
                 →  generate_summary (temp=0)  →  log_to_evals
        │
        ▼
Output: Streamlit UI (answer + tool trace) + Eval logs (Layer 1, 2, 4)
```

---

## Why Hierarchical Chunking?

Standard flat chunking (512-token chunks) forces a tradeoff: small chunks give precise
retrieval but thin LLM context; large chunks give rich context but noisy retrieval.

**Hierarchical chunking solves both sides:**
- **Child chunks (600 chars)** are embedded and searched — small enough for a sharp cosine
  similarity match against the query
- **Parent chunks (6000 chars)** are fetched after the child match — large enough to give
  the LLM complete paragraphs with cause/effect, numbers, and named entities

This is why our faithfulness score reached **4.57/5.0** — the LLM sees full context and
quotes article text directly instead of inferring from thin snippets.

See [Design Document §4.2](DESIGN_DOCUMENT.md#42-hierarchical-chunking-strategy) for the
full comparison with flat `RecursiveCharacterTextSplitter`.

---

## Why Query Rewriting + Reranking?

**Query rewriting** (gpt-5.4-nano, 80 tokens): Converts conversational phrasing like
*"What's going on with OpenAI lately?"* into keyword-dense retrieval queries like
*"OpenAI recent announcements product launches 2026"* — giving the embedding model
high-signal input and boosting relevance.

**Cross-encoder reranking** (ms-marco-MiniLM-L-6-v2): Cosine similarity ranks chunks by
how similar they *sound* to the query. The cross-encoder reads the query and chunk *together*
and scores whether the chunk *answers* the question. This moves the genuinely relevant
article to rank #1 even when several articles share vocabulary.

**Combined result:** MRR = **1.000** (most relevant chunk is always #1), Relevance = **5.00/5.0**.

See [Design Document §6.2.1–6.2.4](DESIGN_DOCUMENT.md#621-quality-impact-hierarchical-chunking)
for the "without vs with" quality analysis.

---

## Eval Results (Regression — June 2026)

| Layer | Metric | Score | Target | Status |
|-------|--------|-------|--------|--------|
| **L1 Retrieval** | Recall@5 | 1.000 | 0.70 | ✅ |
| | Precision@5 | 0.885 | 0.60 | ✅ |
| | MRR | 1.000 | 0.65 | ✅ |
| | NDCG@5 | 1.000 | 0.65 | ✅ |
| **L2 Answer Quality** | Faithfulness | 4.57 | 4.0 | ✅ |
| | Relevance | 5.00 | 4.0 | ✅ |
| | Completeness | 3.86 | 3.5 | ✅ |
| | Conciseness | 4.71 | 3.5 | ✅ |
| **L4 Agent Behaviour** | Tool call accuracy | 0.900 | 85% | ✅ |
| | Avg tool calls/query | 5.4 | 4–6 | ✅ |
| | Broaden trigger rate | 0.667 | 80% | ⚠️ data variability |
| | Hallucination rate | 0.000 | 0% | ✅ |

Full details and per-query breakdown: [Design Document §8](DESIGN_DOCUMENT.md#8-evaluation-report)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM — Agent | `gpt-5.4-mini` (temperature=0 for summaries) |
| LLM — Fast tasks | `gpt-5.4-nano` (query rewrite, eval judge) |
| Embeddings | `text-embedding-3-small` (1536-dim, OpenAI) |
| Vector DB | ChromaDB (local persistent) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (parent + child) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, ~85MB) |
| News Sources | `feedparser`, `newspaper3k`, `newsapi-python` |
| Scheduling | APScheduler (BackgroundScheduler, in-process) |
| UI | Streamlit |
| Eval | LLM-as-Judge (gpt-5.4-nano) + IR metrics (Recall, MRR, NDCG) |
| Language | Python 3.11+ |

---

## Project Structure

```
news-research-assistant/
├── ingestion/
│   ├── fetcher.py          ← RSS + NewsAPI article fetching
│   ├── chunker.py          ← Hierarchical parent+child chunking
│   ├── embedder.py         ← ChromaDB storage + query_collection
│   ├── pipeline.py         ← Full ingestion entry point
│   └── scheduler.py        ← APScheduler wrapper (hourly)
├── agent/
│   ├── tools.py            ← 9 MCP tools (search, rerank, summarise…)
│   └── agent.py            ← Reasoning loop (gpt-5.4-mini)
├── evals/
│   ├── eval_retrieval.py   ← Layer 1: Recall@5, Precision@5, MRR, NDCG
│   ├── eval_answer_quality.py ← Layer 2: LLM-as-judge (4 metrics)
│   ├── eval_agent_behaviour.py ← Layer 4: Tool pattern + hallucination
│   └── results/            ← Run logs (gitignored)
├── ui/
│   └── app.py              ← Streamlit frontend
├── data/
│   └── eval_queries.json   ← 10 labelled evaluation queries
├── scripts/
│   └── generate_eval_logs.py ← Batch agent runner for eval logs
├── DESIGN_DOCUMENT.md      ← Full system design (v2.0)
├── E2E_TEST_GUIDE.md       ← Step-by-step clean-data test guide
├── .env.example            ← API key template
└── requirements.txt
```

---

## Team

| Member | Ownership |
|--------|-----------|
| Member A | Ingestion pipeline — `ingestion/` (fetcher, chunker, embedder, scheduler) |
| Member B | Agent + tools + UI — `agent/`, `ui/` (MCP tools, reasoning loop, Streamlit) |
| Member C | Evaluation — `evals/` (layers 1, 2, 4, eval dataset, scoring harness) |

---

## Quick Start

```bash
# 1. Clone + install
git clone https://github.com/subhanalisha/news-research-assistant.git
cd news-research-assistant
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
pip3 install lxml_html_clean feedparser newsapi-python newspaper3k sentence-transformers

# 2. Add API keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-... and NEWS_API_KEY=...

# 3. Ingest news
python3 ingestion/pipeline.py

# 4. Launch UI
streamlit run ui/app.py

# 5. Run all evals
python3 evals/eval_retrieval.py && \
python3 evals/eval_answer_quality.py && \
python3 evals/eval_agent_behaviour.py
```

For a full clean-data E2E run (clear DB, re-ingest, refresh chunk IDs, run evals):
see [E2E_TEST_GUIDE.md](E2E_TEST_GUIDE.md).

---

## Branch Strategy

```
main      ← stable, demo-ready
dev       ← integration branch
feat/rag      ← Member A
feat/agent    ← Member B
feat/evals    ← Member C
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [DESIGN_DOCUMENT.md](DESIGN_DOCUMENT.md) | Full system design v2.0 — architecture, chunking strategy, tradeoffs, eval report |
| [E2E_TEST_GUIDE.md](E2E_TEST_GUIDE.md) | Step-by-step guide to run a clean-data end-to-end test |
