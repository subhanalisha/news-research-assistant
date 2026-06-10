# End-to-End Test Guide — Clean Data Run

This guide runs the full pipeline from scratch: clear vector DB → ingest fresh news →
run agent → evaluate all 4 layers. Follow every step in order.

---

## Prerequisites

| Requirement | Check command | Expected output |
|-------------|--------------|----------------|
| Python 3.11+ | `python3 --version` | `Python 3.11.x` or higher |
| pip | `pip3 --version` | Any version |
| OpenAI API key | check `.env` | key starts with `sk-` |
| NewsAPI key | check `.env` | 32-char hex string |
| Internet access | `ping google.com` | packets received |

---

## Step 0 — Navigate to project root

```bash
cd news-research-assistant
```

All commands below assume you are in this directory.

---

## Step 1 — Install / verify dependencies

```bash
pip3 install -r requirements.txt
pip3 install lxml_html_clean feedparser newsapi-python newspaper3k sentence-transformers
```

Expected: no errors. Warnings are OK.

---

## Step 2 — Verify API keys in `.env`

```bash
cat .env
```

The file must contain real values (not placeholders):

```
OPENAI_API_KEY=sk-...          # must start with sk-
NEWS_API_KEY=abc123...         # 32-char string from newsapi.org
```

If `.env` does not exist:

```bash
cp .env.example .env
# Then edit .env and fill in your real keys
```

---

## Step 3 — Clear the vector database

This removes all previously ingested articles and chunk embeddings so the
pipeline starts from clean state.

```bash
rm -rf ingestion/chroma_db
```

Verify it is gone:

```bash
ls ingestion/chroma_db 2>/dev/null && echo "NOT cleared" || echo "✅ Cleared"
```

Also clear old eval run logs so stale scores are not mixed in:

```bash
rm -f evals/results/run_*.json
ls evals/results/run_*.json 2>/dev/null && echo "NOT cleared" || echo "✅ Eval logs cleared"
```

---

## Step 4 — Run the ingestion pipeline

Fetches articles from RSS feeds and NewsAPI, chunks them hierarchically,
and embeds + stores them in ChromaDB.

```bash
python3 ingestion/pipeline.py
```

**Expected output (approximately):**

```
[Fetcher] RSS: fetched ~110 articles
[Fetcher] NewsAPI: fetched ~50 articles
[Chunker] 87 articles → 340 parent chunks, 1800 child chunks
[Embedder] Stored 340 parent chunks
[Embedder] Stored 1800 child chunks (embedded)
```

**Verify ChromaDB was populated:**

```bash
python3 -c "
import sys; sys.path.append('ingestion')
from embedder import get_child_collection, get_parent_collection
print('Parent chunks:', get_parent_collection().count())
print('Child chunks: ', get_child_collection().count())
"
```

Expected: parent chunks > 0, child chunks > parent chunks.

---

## Step 5 — Refresh eval_queries.json with new chunk IDs

> ⚠️ **Critical step.** Every time ChromaDB is cleared and re-ingested,
> all chunk UUIDs change. The eval dataset must be updated to match
> the new IDs, otherwise Layer 1 retrieval scores will be 0.

```bash
python3 scripts/refresh_eval_queries.py
```

If that script is not present, refresh manually:

```bash
python3 - <<'EOF'
import sys, json, uuid
sys.path.append('ingestion')
from embedder import query_collection

eval_path = 'data/eval_queries.json'
with open(eval_path) as f:
    dataset = json.load(f)

for item in dataset:
    chunks = query_collection(item['query'], k=5)
    if chunks:
        item['relevant_article_ids'] = [c['chunk_id'] for c in chunks[:3]]
        print(f"Updated: {item['query'][:60]}")
    else:
        print(f"No results for: {item['query'][:60]}")

with open(eval_path, 'w') as f:
    json.dump(dataset, f, indent=2)

print("\n✅ eval_queries.json updated with fresh chunk IDs")
EOF
```

---

## Step 6 — Smoke-test the agent (single query)

Run one query manually to confirm the full agent loop works end-to-end before
running all 10 eval queries.

```bash
cd agent
python3 agent.py
cd ..
```

**Expected output:**

```
==================================================
[Agent] Query: What are the latest AI developments from OpenAI?
[Agent] → Calling: rewrite_query(...)
[Agent] → Calling: search_news(...)
[Agent] → Calling: rerank_chunks(...)
[Agent] → Calling: generate_summary(...)
[Agent] → Calling: log_to_evals(...)

[Agent] Final answer:
OpenAI announced... [TechCrunch, 2026-06-...]
```

If you see errors, check [Troubleshooting](#troubleshooting) at the bottom.

---

## Step 7 — Generate eval logs for all 10 queries

Runs the agent against every query in `data/eval_queries.json` and saves
run logs to `evals/results/`. The `--clean` flag removes any leftover logs first.

```bash
python3 scripts/generate_eval_logs.py --clean
```

**Expected output:**

```
[Cleanup] Removed 0 old run logs.
Running agent for 10 eval queries...

[1/10] [recency       ] What are the latest AI developments from OpenAI?
  ✅ 7 tool calls: ['rewrite_query', 'search_news', ...]
[2/10] [standard      ] What was announced at Apple WWDC 2026?
  ✅ 6 tool calls: [...]
...
Done — 10 passed, 0 failed
```

Verify 10 logs were created:

```bash
ls evals/results/run_*.json | wc -l
# Expected: 10
```

---

## Step 8 — Run Layer 1: Retrieval Evals

Measures Recall@5, Precision@5, MRR, NDCG@5 against the labelled chunk IDs.

```bash
python3 evals/eval_retrieval.py
```

**Expected passing scores:**

```
========================================
Layer 1 — Retrieval Eval Results
========================================
Metric             Score   Target   Status
----------------------------------------
Recall@5           ≥0.70    0.70   ✅ Pass
Precision@5        ≥0.60    0.60   ✅ Pass
MRR                ≥0.65    0.65   ✅ Pass
NDCG@5             ≥0.65    0.65   ✅ Pass
```

---

## Step 9 — Run Layer 2: Answer Quality Evals

LLM-as-judge (gpt-5.4-nano) scores each run log on 4 metrics (1–5 scale).

```bash
python3 evals/eval_answer_quality.py
```

**Expected passing scores:**

```
=============================================
Layer 2 — Answer Quality Eval Results
=============================================
Metric              Score   Target   Status
---------------------------------------------
faithfulness        ≥4.0     4.0    ✅ Pass
relevance           ≥4.0     4.0    ✅ Pass
completeness        ≥3.5     3.5    ✅ Pass
conciseness         ≥3.5     3.5    ✅ Pass
```

> Note: LLM-as-judge scores have natural variance (±0.3) between runs.
> Scores just below target on a single run are normal; re-run once to confirm.

---

## Step 10 — Run Layer 4: Agent Behaviour Evals

Checks tool call patterns, broaden trigger rate, and hallucination rate.

```bash
python3 evals/eval_agent_behaviour.py
```

**Expected passing scores:**

```
=======================================================
Layer 4 — Agent Behaviour Eval Results
=======================================================
Metric                          Score   Target   Status
-------------------------------------------------------
tool_call_accuracy              ≥0.85    85%    ✅ Pass
avg_tool_calls_per_query        4-6      4-6    ✅ Pass
broaden_trigger_rate            ≥0.80    80%    ✅ Pass
hallucination_rate              0.00     0%     ✅ Pass
```

---

## Step 11 — Run all evals in one command

```bash
python3 evals/eval_retrieval.py && \
python3 evals/eval_answer_quality.py && \
python3 evals/eval_agent_behaviour.py
```

---

## Step 12 — Launch the Streamlit UI (optional)

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`. Type a question and verify:
- Answer appears with inline citations `[Source, Date]`
- Tool trace panel shows all tool calls
- Source cards appear at the bottom

---

## Full E2E Command Sequence (copy-paste)

```bash
# 0. Go to project root
cd news-research-assistant

# 1. Install dependencies
pip3 install -r requirements.txt
pip3 install lxml_html_clean feedparser newsapi-python newspaper3k sentence-transformers

# 2. Clear vector DB and old eval logs
rm -rf ingestion/chroma_db
rm -f evals/results/run_*.json

# 3. Ingest fresh news
python3 ingestion/pipeline.py

# 4. Refresh eval_queries.json with new chunk IDs
python3 - <<'EOF'
import sys, json
sys.path.append('ingestion')
from embedder import query_collection
eval_path = 'data/eval_queries.json'
with open(eval_path) as f:
    dataset = json.load(f)
for item in dataset:
    chunks = query_collection(item['query'], k=5)
    if chunks:
        item['relevant_article_ids'] = [c['chunk_id'] for c in chunks[:3]]
with open(eval_path, 'w') as f:
    json.dump(dataset, f, indent=2)
print("✅ eval_queries.json refreshed")
EOF

# 5. Generate eval logs (runs agent for all 10 queries)
python3 scripts/generate_eval_logs.py --clean

# 6. Run all eval layers
python3 evals/eval_retrieval.py
python3 evals/eval_answer_quality.py
python3 evals/eval_agent_behaviour.py
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: feedparser` | Missing package | `pip3 install feedparser newsapi-python newspaper3k` |
| `ModuleNotFoundError: lxml_html_clean` | Missing package | `pip3 install lxml_html_clean` |
| `openai.AuthenticationError: 401` | Wrong/placeholder API key | Edit `.env`, set real `OPENAI_API_KEY` |
| `NewsAPI error: apiKeyInvalid` | Wrong NewsAPI key | Edit `.env`, set real `NEWS_API_KEY` |
| `Layer 1 all zeros (0.000)` | Stale chunk IDs in eval_queries.json | Re-run Step 5 (refresh eval_queries.json) |
| `[Evals] No run logs found` | Step 7 not done | Run `python3 scripts/generate_eval_logs.py --clean` |
| `chromadb empty after pipeline` | Wrong working directory | Run all commands from `news-research-assistant/` root |
| Cross-encoder slow on first run | Downloading model (~85MB) | Wait once; cached after first run |
| `filter_by_date unexpected keyword` | Old log format | Re-run `generate_eval_logs.py --clean` |

---

*Last updated: June 2026*
