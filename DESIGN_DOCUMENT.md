# News Research Assistant — System Design Document

**Project:** Capstone Project 06 — RAG + MCP + Agents + Evals  
**Team Size:** 3 Members  
**Timeline:** 7-Day Sprint  
**GitHub:** https://github.com/subhanalisha/news-research-assistant  
**Date:** June 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Block Diagram](#3-block-diagram)
4. [Detailed Design — Component Flows](#4-detailed-design--component-flows)
   - 4.1 [Data Ingestion Flow](#41-data-ingestion-flow)
   - 4.2 [Hierarchical Chunking Strategy](#42-hierarchical-chunking-strategy)
   - 4.3 [Agent Reasoning Loop (MCP)](#43-agent-reasoning-loop-mcp)
   - 4.4 [Evaluation Framework (4-Layer)](#44-evaluation-framework-4-layer)
   - 4.5 [Streamlit UI Flow](#45-streamlit-ui-flow)
5. [Data Processing](#5-data-processing)
6. [System Design — Architecture & Tradeoffs](#6-system-design--architecture--tradeoffs)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Steps to Run the Project](#8-steps-to-run-the-project)

---

## 1. Problem Statement

### 1.1 Scoping

Journalists, researchers, and analysts spend significant time manually sifting through hundreds of news sources to find relevant, accurate, and up-to-date information on a given topic. Current news aggregators:

- Return keyword-matched snippets with no contextual synthesis
- Do not allow multi-step query refinement
- Provide no source attribution or citation tracking
- Cannot handle compound questions spanning multiple articles

### 1.2 Clarity

**Core Problem:** There is no intelligent, agent-driven system that can:

1. Continuously ingest fresh news from multiple sources
2. Retrieve the most contextually relevant information using semantic search
3. Reason over that information through a multi-step agent loop
4. Produce grounded, cited answers with verifiable sources
5. Self-evaluate answer quality automatically

**Target Users:** Researchers, journalists, investment analysts, product teams tracking competitive intelligence.

**Scope Boundaries:**

| In Scope | Out of Scope |
|----------|-------------|
| English-language news articles | Multi-language support |
| Tech, AI, startup news topics | Financial data / real-time stock prices |
| Text-based retrieval and answers | Image/video news retrieval |
| Automated quality evaluation | Human-in-the-loop review UI |
| 5 RSS sources + NewsAPI | Social media monitoring |

---

## 2. Solution Overview

The **News Research Assistant** is a Retrieval-Augmented Generation (RAG) system enhanced with an agentic reasoning loop built on the **Model Context Protocol (MCP)** tool pattern.

### 2.1 Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| LLM (Reasoning) | `gpt-5.4-mini` | Best cost-to-capability ratio for agent loop |
| LLM (Fast Tasks) | `gpt-5.4-nano` | Cheaper for query rewriting and eval scoring |
| Embeddings | `text-embedding-3-small` | OpenAI native, strong semantic accuracy |
| Vector DB | ChromaDB (local) | Zero-ops, file-based, fast for prototyping |
| Chunking | Hierarchical (Parent-Child) | Small chunks for precise retrieval, large chunks for rich LLM context |
| Reranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Improves relevance ordering beyond cosine similarity |
| News Sources | RSS Feeds + NewsAPI | Breadth (RSS) + targeted query coverage (NewsAPI) |
| Scheduling | APScheduler | Lightweight, runs in-process |

### 2.2 Solution Capabilities

- 🔄 **Auto-ingestion**: Scheduled ingestion from 5 RSS feeds and NewsAPI every hour
- 🧠 **Semantic retrieval**: Child chunk vector search with parent context expansion
- 🔍 **Query rewriting**: LLM rewrites user queries into keyword-rich search terms before retrieval
- 📊 **Cross-encoder reranking**: Precision reordering of retrieved chunks
- 🤖 **Agentic reasoning**: 9-tool MCP agent that decides dynamically which tools to call
- 📝 **Cited answers**: Every factual claim is attributed to source + date
- 📈 **Auto-evaluation**: 4-layer eval framework that scores every agent run

---

## 3. Block Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         NEWS RESEARCH ASSISTANT                              ║
║                         System Architecture                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                        DATA SOURCES (External)                          │
 │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
 │  │ TechCrunch  │  │   Wired     │  │  The Verge   │  │  Ars Tech   │  │
 │  │  RSS Feed   │  │  RSS Feed   │  │  RSS Feed    │  │  RSS Feed   │  │
 │  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  │
 │         └────────────────┴─────────────────┴─────────────────┘          │
 │                              │                                           │
 │                    ┌─────────▼──────────┐                                │
 │                    │   NewsAPI Client   │                                │
 │                    │ (OpenAI, AI, Tech) │                                │
 │                    └─────────┬──────────┘                                │
 └──────────────────────────────┼──────────────────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                    INGESTION PIPELINE  (APScheduler)                     │
 │                                                                          │
 │   fetcher.py        chunker.py            embedder.py                   │
 │  ┌──────────┐      ┌─────────────────┐   ┌──────────────────────────┐  │
 │  │ Fetch &  │─────▶│ Hierarchical    │──▶│ Embed child chunks with  │  │
 │  │ Parse    │      │ Chunking        │   │ text-embedding-3-small   │  │
 │  │ Articles │      │ Parent: 1500tok │   │                          │  │
 │  └──────────┘      │ Child:   150tok │   │ Store in ChromaDB:       │  │
 │                    └─────────────────┘   │  ├─ news_child_chunks    │  │
 │                                          │  └─ news_parent_chunks   │  │
 │                                          └──────────────────────────┘  │
 └──────────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │  ChromaDB (Local)    │
                    │  ┌────────────────┐  │
                    │  │ child_chunks   │  │ ← Embedded (searchable)
                    │  │ (vectors)      │  │
                    │  ├────────────────┤  │
                    │  │ parent_chunks  │  │ ← Text only (context store)
                    │  │ (text store)   │  │
                    │  └────────────────┘  │
                    └───────────┬──────────┘
                                │
 ┌──────────────────────────────▼───────────────────────────────────────────┐
 │                  AGENT REASONING LOOP  (agent.py)                        │
 │                  Model: gpt-5.4-mini  │  Protocol: MCP                   │
 │                                                                           │
 │   User Query                                                              │
 │       │                                                                   │
 │       ▼                                                                   │
 │  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐                 │
 │  │rewrite_query│───▶│ search_news │───▶│rerank_chunks │                 │
 │  │(gpt-5.4-nano│    │ (ChromaDB   │    │(cross-encoder│                 │
 │  │  query opt) │    │  semantic   │    │ ms-marco     │                 │
 │  └─────────────┘    │  retrieval) │    │ MiniLM-L6)   │                 │
 │                     └──────┬──────┘    └──────┬───────┘                 │
 │                            │                   │                         │
 │              ┌─────────────▼─────┐             │                        │
 │              │  Thin? (<3 chunks)│             │                        │
 │              │  → broaden_search │             │                        │
 │              │  Recency query?   │             │                        │
 │              │  → filter_by_date │             │                        │
 │              └───────────────────┘             │                        │
 │                                                ▼                         │
 │                                    ┌───────────────────┐                 │
 │                                    │ generate_summary  │                 │
 │                                    │ (gpt-5.4-mini,    │                 │
 │                                    │  cited answer)    │                 │
 │                                    └─────────┬─────────┘                 │
 │                                              │                            │
 │                                    ┌─────────▼─────────┐                 │
 │                                    │   log_to_evals    │                 │
 │                                    │ (persist run log) │                 │
 │                                    └───────────────────┘                 │
 └──────────────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────▼──────────────────────┐
              │         OUTPUT LAYER                    │
              │  ┌───────────────┐  ┌────────────────┐ │
              │  │ Streamlit UI  │  │  Eval Framework│ │
              │  │ Answer +      │  │  Layer 1-4     │ │
              │  │ Tool Trace    │  │  Auto-scoring  │ │
              │  └───────────────┘  └────────────────┘ │
              └────────────────────────────────────────┘
```

---

## 4. Detailed Design — Component Flows

### 4.1 Data Ingestion Flow

```
Trigger: APScheduler (every 60 min) or manual run of pipeline.py
                                │
                ┌───────────────▼────────────────┐
                │         fetcher.py              │
                │                                 │
                │  1. Parse 5 RSS feeds           │
                │     feedparser → entry list     │
                │                                 │
                │  2. NewsAPI queries (3 terms):  │
                │     "OpenAI"         (20 art.)  │
                │     "artificial intel" (20 art.)│
                │     "technology startups"(10)   │
                │                                 │
                │  3. newspaper3k full-text fetch │
                │     per article URL             │
                └───────────────┬────────────────┘
                                │  ~70-100 articles/run
                ┌───────────────▼────────────────┐
                │         chunker.py              │
                │                                 │
                │  Per article:                   │
                │  ┌──────────────────────────┐   │
                │  │ Parent splitter          │   │
                │  │ chunk_size = 6000 chars  │   │
                │  │ overlap    =  400 chars  │   │
                │  │ separators: \n\n \n . sp │   │
                │  └──────────┬───────────────┘   │
                │             │ N parent chunks    │
                │  ┌──────────▼───────────────┐   │
                │  │ Child splitter per parent │   │
                │  │ chunk_size =  600 chars  │   │
                │  │ overlap    =   80 chars  │   │
                │  │ Child stores: parent_id  │   │
                │  └──────────────────────────┘   │
                └───────────────┬────────────────┘
                                │
                ┌───────────────▼────────────────┐
                │         embedder.py             │
                │                                 │
                │  Parent chunks →                │
                │    news_parent_chunks (text)    │
                │    Stored as metadata only      │
                │                                 │
                │  Child chunks →                 │
                │    news_child_chunks (vectors)  │
                │    OpenAI embed (1536-dim)      │
                │    Upserted with chunk_id key   │
                └────────────────────────────────┘

Output: ChromaDB persisted at ingestion/chroma_db/
```

### 4.2 Hierarchical Chunking Strategy

**Why Hierarchical?**

Standard flat chunking creates a tradeoff: small chunks give precise retrieval but poor LLM context; large chunks give rich context but noisy retrieval. Hierarchical chunking solves both sides.

```
Article (full text ~3000-8000 chars)
│
├─ Parent Chunk 0  (1500 tokens ≈ 6000 chars)
│    chunk_id: "abc-123"
│    ├─ Child Chunk 0  (150 tokens ≈ 600 chars)  parent_id: "abc-123"
│    ├─ Child Chunk 1  (150 tokens ≈ 600 chars)  parent_id: "abc-123"
│    └─ Child Chunk 2  (150 tokens ≈ 600 chars)  parent_id: "abc-123"
│
├─ Parent Chunk 1  (1500 tokens ≈ 6000 chars)
│    chunk_id: "def-456"
│    ├─ Child Chunk 3  parent_id: "def-456"
│    └─ Child Chunk 4  parent_id: "def-456"
│
└─ Parent Chunk 2 ...

At query time:
  1. Embed query → search news_child_chunks (small = precise match)
  2. For each matched child → fetch parent chunk by parent_id
  3. Return parent chunk text to LLM (large = rich context)
```

**Retrieval with Parent Expansion (embedder.py `query_collection`):**
```
User query ──▶ embed ──▶ cosine similarity ──▶ top-k child chunks
                                                      │
                                         fetch parent_id from metadata
                                                      │
                                         ChromaDB.get(parent_collection)
                                                      │
                                         Return parent chunk text
                                         (contains full paragraph context)
```

### 4.3 Agent Reasoning Loop (MCP)

The agent follows the **Model Context Protocol (MCP)** tool-call pattern — a structured way for LLMs to invoke named tools with typed inputs/outputs.

#### 9 MCP Tools

| Tool | Model | Purpose |
|------|-------|---------|
| `rewrite_query` | gpt-5.4-nano | Optimize user query for retrieval |
| `search_news` | ChromaDB | Semantic vector search (k=6) |
| `filter_by_date` | — | Restrict to last N days |
| `filter_by_source` | — | Restrict to specific outlets |
| `fetch_full_article` | ChromaDB | Get all chunks for one article |
| `broaden_search` | ChromaDB | Re-search with k=12 when thin |
| `rerank_chunks` | CrossEncoder | Precision reordering |
| `generate_summary` | gpt-5.4-mini | Produce cited answer |
| `log_to_evals` | — | Persist run log for evaluation |

#### Standard Query Flow (most common path)

```
User: "What are the latest AI developments from OpenAI?"
│
▼ TURN 1 — Agent calls:
  rewrite_query("What are the latest AI developments from OpenAI?")
  → "OpenAI latest announcements GPT model release 2026"
│
▼ TURN 2 — Agent calls:
  search_news("OpenAI latest announcements GPT model release 2026", k=6)
  → 6 parent chunks returned
│
▼ TURN 3 — Agent evaluates: query mentions "latest" → date filter
  filter_by_date(chunks, days=3)
  → filtered to recent chunks
│
▼ TURN 4 — Agent calls:
  rerank_chunks("OpenAI latest announcements...", chunks)
  → chunks reordered by cross-encoder score
│
▼ TURN 5 — Agent calls:
  generate_summary("What are the latest AI developments...", chunks)
  → cited answer: "OpenAI announced... [TechCrunch, 2026-06-08]"
│
▼ TURN 6 — Agent calls:
  log_to_evals(query, chunks, answer, tool_trace)
  → saved to evals/results/run_20260610_143022.json
│
▼ Agent stops (finish_reason: "stop")
  Final answer returned to user
```

#### Thin Retrieval Flow

```
search_news returns < 3 chunks
│
▼ Agent calls: broaden_search(query, k=12)
  → tries wider search with doubled k
│
▼ → rerank_chunks → generate_summary
```

#### Tool Call Decision Rules (System Prompt)

1. Always call `rewrite_query` first
2. Call `search_news` with rewritten query
3. If fewer than 3 chunks → call `broaden_search`
4. If query contains "latest/today/recent" → call `filter_by_date(days=3)`
5. Always call `rerank_chunks` after retrieval
6. Only call `generate_summary` when 3+ chunks available
7. Never answer from memory — only use retrieved chunks
8. Always call `log_to_evals` after generating answer

### 4.4 Evaluation Framework (4-Layer)

Every agent run is automatically scored across 4 evaluation layers.

#### Layer 1 — Retrieval Quality (`eval_retrieval.py`)

```
Eval dataset: data/eval_queries.json (10 queries with labelled chunk IDs)
│
For each query:
  query_collection(query, k=5) → retrieved chunk IDs
  Compare vs. relevant_article_ids (parent chunk IDs) from eval_queries.json
│
Metrics computed:
  Recall@5    = hits / total relevant
  Precision@5 = hits / 5 retrieved
  MRR         = 1/rank of first relevant result
  NDCG@5      = normalized discounted cumulative gain

Targets: Recall≥0.70, Precision≥0.60, MRR≥0.65, NDCG≥0.65
```

#### Layer 2 — Answer Quality (`eval_answer_quality.py`)

```
For each run log in evals/results/:
  Load: query, answer, chunks_retrieved
  │
  Call gpt-5.4-nano as judge, 4 metrics × 1-5 scale:
  ┌─────────────┬──────────────────────────────────────────┐
  │ Faithfulness│ Every claim backed by retrieved articles  │
  │ Relevance   │ Answer addresses what was asked           │
  │ Completeness│ All key article points reflected          │
  │ Conciseness │ No unnecessary padding                    │
  └─────────────┴──────────────────────────────────────────┘

Targets: Faithfulness≥4.0, Relevance≥4.0,
         Completeness≥3.5, Conciseness≥3.5
```

#### Layer 3 — Source Quality (`eval_source_quality.py`)

```
For each run log:
  - Source diversity: count of unique sources per answer
  - Recency score: fraction of chunks from last 7 days
  - URL validity: check source field not empty

Target: Diversity≥2 sources, Recency≥60%
```

#### Layer 4 — Agent Behaviour (`eval_agent_behaviour.py`)

```
For each run log vs. eval dataset pattern:
  ┌─────────────────────────┬──────────────────────────────────────────┐
  │ tool_call_accuracy      │ Did agent call all expected tools?        │
  │                         │ standard:       rewrite→search→rerank→sum│
  │                         │ recency:        +filter_by_date          │
  │                         │ thin_retrieval: +broaden_search          │
  ├─────────────────────────┼──────────────────────────────────────────┤
  │ avg_tool_calls/query    │ Mean tool calls per run (target: 4-6)    │
  ├─────────────────────────┼──────────────────────────────────────────┤
  │ broaden_trigger_rate    │ Was broaden_search called for thin queries│
  │                         │ (target: ≥80%)                           │
  ├─────────────────────────┼──────────────────────────────────────────┤
  │ hallucination_rate      │ generate_summary called before search?   │
  │                         │ (target: 0%)                             │
  └─────────────────────────┴──────────────────────────────────────────┘
```

### 4.5 Streamlit UI Flow

```
User opens http://localhost:8501
│
├─ Sidebar: Configuration
│   └─ date filter toggle, source filter, k value
│
├─ Main: Text input → "Ask a question about recent news"
│
├─ On submit:
│   ├─ Calls run_agent(query, verbose=True)
│   ├─ Streams tool trace in expandable panel
│   │   └─ Each tool call shown: name + inputs + result preview
│   ├─ Displays final cited answer
│   └─ Displays source cards (title, source, date, URL)
│
└─ Footer: Session run count, last ingestion timestamp
```

---

## 5. Data Processing

### 5.1 Data Sources

| Source | Type | Volume/Run | Topics |
|--------|------|-----------|--------|
| TechCrunch RSS | RSS Feed | ~30 articles | Tech, Startups, AI |
| Wired RSS | RSS Feed | ~20 articles | Tech culture, Science |
| The Verge RSS | RSS Feed | ~25 articles | Consumer tech |
| Ars Technica RSS | RSS Feed | ~20 articles | Deep tech, Science |
| VentureBeat RSS | RSS Feed | ~15 articles | AI, Enterprise tech |
| NewsAPI — "OpenAI" | REST API | 20 articles | OpenAI news |
| NewsAPI — "artificial intelligence" | REST API | 20 articles | AI ecosystem |
| NewsAPI — "technology startups" | REST API | 10 articles | VC, funding |

**Total:** ~160 articles per ingestion run

### 5.2 Handling PII

News articles are public journalistic content and do not contain private user data. However:

- **No user queries stored to disk** unless explicitly submitted for eval logging
- **Eval run logs** contain only the query text and retrieved public article excerpts
- **API keys** are stored in `.env` (gitignored), never committed to source control
- The `.env.example` file contains only placeholder values

```
.env           ← GITIGNORED (real keys)
.env.example   ← Committed (placeholder values only)
.gitignore     ← Explicitly excludes .env, chroma_db/, __pycache__/
```

### 5.3 Guardrails

| Guardrail | Implementation |
|-----------|---------------|
| No hallucination | System prompt rule: "Never answer from memory. Only use retrieved chunks." |
| Citation required | generate_summary prompt: "Cite every claim inline as [Source, Date]" |
| Empty retrieval guard | generate_summary returns explicit "no relevant articles" message if chunks=[] |
| Thin retrieval guard | Agent rule: broaden_search triggered if <3 chunks returned |
| Tool order enforcement | System prompt mandates: rewrite→search→rerank→summarise ordering |
| Eval logging | Every answer automatically logged with tool trace for auditability |
| API key safety | All keys via `os.getenv()` from dotenv, never hardcoded |

---

## 6. System Design — Architecture & Tradeoffs

### 6.1 Architecture Summary

```
┌─────────────────────────────────────────────────────┐
│                 Technology Stack                     │
├───────────────┬─────────────────────────────────────┤
│ Layer         │ Technology                           │
├───────────────┼─────────────────────────────────────┤
│ LLM (Agent)   │ OpenAI gpt-5.4-mini                 │
│ LLM (Fast)    │ OpenAI gpt-5.4-nano                 │
│ Embeddings    │ text-embedding-3-small (1536-dim)    │
│ Vector Store  │ ChromaDB (local persistent)          │
│ Reranking     │ cross-encoder/ms-marco-MiniLM-L-6-v2│
│ Scheduling    │ APScheduler (BackgroundScheduler)    │
│ News Sources  │ feedparser, newspaper3k, newsapi     │
│ Chunking      │ LangChain RecursiveCharacterSplitter │
│ UI            │ Streamlit                            │
│ Eval          │ LLM-as-Judge (gpt-5.4-nano) + IR     │
│ Language      │ Python 3.11+                         │
└───────────────┴─────────────────────────────────────┘
```

### 6.2 Architectural Tradeoffs

| Decision | Alternative Considered | Why We Chose Current |
|----------|----------------------|---------------------|
| ChromaDB local | Pinecone / Weaviate cloud | Zero infra cost, no latency overhead, good for 10k-100k chunks |
| Hierarchical chunking | Flat 512-token chunks | Parent context gives LLM richer grounding; child chunks give retrieval precision |
| Cross-encoder reranking | Cosine score only | Cross-encoder scores semantic relevance of (query, chunk) pairs jointly — more accurate than embedding distance alone |
| gpt-5.4-nano for eval judge | gpt-5.4-mini | Eval runs thousands of calls; nano cuts cost 5-10x with acceptable scoring quality |
| MCP tool pattern | LangChain AgentExecutor | MCP gives full control of tool schemas and execution without framework overhead |
| APScheduler in-process | Celery / Airflow | No broker needed, suitable for single-machine prototype |
| RSS + NewsAPI | Web scraping | RSS is structured and legal; NewsAPI provides topic-targeted search without crawling |

### 6.3 Data Flow Summary

```
External News ──▶ Fetch ──▶ Chunk ──▶ Embed ──▶ ChromaDB
                                                    │
User Query ──▶ Rewrite ──▶ Vector Search ──▶ Parent Expand
                                                    │
                     Cross-Encoder Rerank ◀──────────
                                │
                    Optional: Date Filter / Broaden
                                │
                    LLM Generate Cited Answer
                                │
              ┌─────────────────┴───────────────────┐
              │                                     │
         Streamlit UI                        Eval Log (JSON)
         (Answer + Trace)                    (Layer 1-4 scoring)
```

---

## 7. Evaluation Metrics

### 7.1 Metric Definitions & Targets

| Layer | Metric | Formula | Target | Status |
|-------|--------|---------|--------|--------|
| **L1 Retrieval** | Recall@5 | relevant_retrieved / total_relevant | ≥ 0.70 | |
| | Precision@5 | relevant_retrieved / 5 | ≥ 0.60 | |
| | MRR | 1/rank_of_first_hit | ≥ 0.65 | |
| | NDCG@5 | Weighted discounted gain | ≥ 0.65 | |
| **L2 Answer** | Faithfulness | LLM judge 1-5 | ≥ 4.0 | |
| | Relevance | LLM judge 1-5 | ≥ 4.0 | |
| | Completeness | LLM judge 1-5 | ≥ 3.5 | |
| | Conciseness | LLM judge 1-5 | ≥ 3.5 | |
| **L3 Source** | Source diversity | unique_sources / answer | ≥ 2 | |
| | Recency | chunks_from_last_7d / total | ≥ 60% | |
| **L4 Agent** | Tool call accuracy | correct_tool_pattern / total | ≥ 85% | |
| | Avg tool calls | mean calls/query | 4-6 | |
| | Broaden trigger rate | broaden_triggered / thin_queries | ≥ 80% | |
| | Hallucination rate | summary_before_search / total | 0% | |

### 7.2 Eval Dataset

`data/eval_queries.json` — 10 labelled queries:

| # | Query | Pattern | Key Topics |
|---|-------|---------|-----------|
| 1 | Latest AI developments from OpenAI? | recency | Greg Brockman, OpenAI IPO |
| 2 | What was announced at Apple WWDC 2026? | standard | WWDC announcements |
| 3 | Recent AI startup funding rounds? | recency | Jack Altman, Seed 100 |
| 4 | Latest news about Google I/O? | standard | Google I/O |
| 5 | Quantum computing breakthroughs? | thin_retrieval | UK AI supercomputer |
| 6 | Autonomous vehicles and robotics? | thin_retrieval | Uber self-driving |
| 7 | Greg Brockman at OpenAI recently? | standard | Brockman product control |
| 8 | macOS 27 and Apple Silicon? | standard | Intel dropped |
| 9 | Very latest AI news from today? | recency | filter_by_date trigger |
| 10 | Space exploration and satellites? | thin_retrieval | broaden_search trigger |

### 7.3 Task-Specific Evals

- **Recency queries** (3/10): Must trigger `filter_by_date(days=3)` — validates temporal awareness
- **Thin retrieval** (3/10): Must trigger `broaden_search` — validates graceful degradation
- **Standard** (4/10): Must use `rewrite_query → search → rerank → summarise` — validates base flow

---

## 8. Steps to Run the Project

### 8.1 Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.11+ | `python3 --version` |
| pip | 23+ | `pip3 --version` |
| Git | Any | `git --version` |
| OpenAI API Key | Active | platform.openai.com |
| NewsAPI Key | Active | newsapi.org |
| ~2 GB disk | For ChromaDB + models | |
| Internet | For ingestion | |

### 8.2 Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/subhanalisha/news-research-assistant.git
cd news-research-assistant

# 2. Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip3 install -r requirements.txt

# 4. Install extra NLP dependencies
pip3 install lxml_html_clean feedparser newsapi-python newspaper3k
pip3 install sentence-transformers  # for cross-encoder reranking
```

### 8.3 Configure API Keys

```bash
# 5. Copy the example env file
cp .env.example .env

# 6. Edit .env with your real keys
nano .env   # or open in any editor
```

Your `.env` file should contain:
```
OPENAI_API_KEY=sk-your-openai-key-here
NEWS_API_KEY=your-newsapi-key-here
```

> ⚠️ **Security**: Never commit `.env` to GitHub. The `.gitignore` already excludes it.

### 8.4 Run the Ingestion Pipeline

```bash
# 7. Run the ingestion pipeline (fetches articles, chunks, and embeds)
cd news-research-assistant
python3 ingestion/pipeline.py
```

Expected output:
```
[Fetcher] RSS: fetched 110 articles
[Fetcher] NewsAPI: fetched 50 articles
[Chunker] 87 articles → 342 parent chunks, 1820 child chunks
[Embedder] Stored 342 parent chunks
[Embedder] Embedded and stored 1820 child chunks
```

### 8.5 Run the Agent

```bash
# 8. Test the agent directly
cd agent
python3 agent.py
```

Expected output:
```
==================================================
[Agent] Query: What are the latest AI developments from OpenAI?
[Agent] → Calling: rewrite_query(...)
[Agent] → Calling: search_news(...)
[Agent] → Calling: filter_by_date(...)
[Agent] → Calling: rerank_chunks(...)
[Agent] → Calling: generate_summary(...)
[Agent] → Calling: log_to_evals(...)

[Agent] Final answer:
OpenAI announced... [TechCrunch, 2026-06-08]. Greg Brockman...
```

### 8.6 Generate Eval Logs (for all 10 queries)

```bash
# 9. Run agent against all eval queries to populate evals/results/
cd news-research-assistant
python3 scripts/generate_eval_logs.py
```

### 8.7 Run Evaluation Layers

```bash
# 10. Layer 1 — Retrieval quality (Recall, Precision, MRR, NDCG)
python3 evals/eval_retrieval.py

# 11. Layer 2 — Answer quality (Faithfulness, Relevance, etc.)
python3 evals/eval_answer_quality.py

# 12. Layer 3 — Source quality
python3 evals/eval_source_quality.py

# 13. Layer 4 — Agent behaviour
python3 evals/eval_agent_behaviour.py
```

### 8.8 Launch the Streamlit UI

```bash
# 14. Launch the web interface
cd news-research-assistant
streamlit run ui/app.py
# Opens at http://localhost:8501
```

### 8.9 Run Scheduled Ingestion (optional)

```bash
# 15. Start the scheduler (re-ingests every 60 minutes)
python3 ingestion/scheduler.py
```

### 8.10 Project File Structure

```
news-research-assistant/
├── .env                        ← API keys (GITIGNORED)
├── .env.example                ← Template (safe to commit)
├── .gitignore
├── requirements.txt
├── DESIGN_DOCUMENT.md          ← This document
│
├── ingestion/
│   ├── fetcher.py              ← RSS + NewsAPI article fetching
│   ├── chunker.py              ← Hierarchical chunking (parent+child)
│   ├── embedder.py             ← ChromaDB storage + query_collection
│   ├── pipeline.py             ← Full ingestion pipeline entry point
│   ├── scheduler.py            ← APScheduler wrapper
│   └── chroma_db/              ← Persisted vector DB (GITIGNORED)
│
├── agent/
│   ├── tools.py                ← 9 MCP tools implementation
│   └── agent.py                ← Agentic reasoning loop (gpt-5.4-mini)
│
├── evals/
│   ├── eval_retrieval.py       ← Layer 1: Recall, Precision, MRR, NDCG
│   ├── eval_answer_quality.py  ← Layer 2: LLM-as-judge scoring
│   ├── eval_source_quality.py  ← Layer 3: Source diversity + recency
│   ├── eval_agent_behaviour.py ← Layer 4: Tool call pattern analysis
│   └── results/                ← Run logs (GITIGNORED)
│
├── data/
│   └── eval_queries.json       ← 10 labelled evaluation queries
│
├── ui/
│   └── app.py                  ← Streamlit interface
│
└── scripts/
    └── generate_eval_logs.py   ← Batch agent runner for eval logs
```

### 8.11 Common Issues & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: feedparser` | `pip3 install feedparser newsapi-python newspaper3k` |
| `ImportError: lxml_html_clean` | `pip3 install lxml_html_clean` |
| `401 Unauthorized` OpenAI | Check `.env` has real key, not placeholder |
| `All Layer 1 scores = 0` | Re-run `generate_eval_logs.py` after re-ingestion (chunk IDs change) |
| `No run logs found` | Run `scripts/generate_eval_logs.py` before running evals |
| `ChromaDB empty` | Run `python3 ingestion/pipeline.py` first |
| Cross-encoder slow first run | Downloading `ms-marco-MiniLM-L-6-v2` model (~85MB) — wait once |

---

## Team Responsibilities

| Member | Ownership |
|--------|-----------|
| Member A | Ingestion pipeline — `ingestion/` (fetcher, chunker, embedder, scheduler) |
| Member B | Agent + tools — `agent/` (tools.py, agent.py, MCP schema, Streamlit UI) |
| Member C | Evaluation framework — `evals/` (4 layers, eval dataset, scoring) |

---

*Document version 1.0 — Generated June 2026*  
*GitHub: https://github.com/subhanalisha/news-research-assistant*
