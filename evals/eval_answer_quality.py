"""
Member C — Layer 2: Answer Quality Evals (LLM-as-Judge)
Measures: Faithfulness, Relevance, Completeness, Conciseness
"""

import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

METRICS = ["faithfulness", "relevance", "completeness", "conciseness"]

# Chain-of-thought judge prompt — model reasons before scoring
JUDGE_PROMPT = """You are an expert evaluator for a news RAG (Retrieval-Augmented Generation) system.
Score the answer on {metric_upper} using a 1–5 scale.

IMPORTANT CONTEXT:
- This is a RAG system with LIMITED retrieved articles (sometimes only 1-3 articles).
- Score ONLY based on how well the answer uses the provided articles, NOT on whether the answer
  seems "complete" by some external standard.
- A concise answer that covers everything in 2 articles should score 5 on completeness.
- Do NOT penalise for information not present in the retrieved articles.

SCALE:
5 = {score_5}
4 = {score_4}
3 = {score_3}
2 = {score_2}
1 = {score_1}

Question: {query}

Retrieved Articles (these are ALL the sources the system had access to):
{chunks}

Answer to evaluate:
{answer}

Instructions:
1. Think step by step (2-3 sentences): what information is in the articles, and does the answer use it?
2. Final line MUST be valid JSON: {{"score": <1-5>, "reason": "<one sentence>"}}

Your evaluation:"""

METRIC_RUBRICS = {
    "faithfulness": {
        "score_5": "Every claim is directly traceable to a specific sentence in the articles. Perfect grounding.",
        "score_4": "Core claims are all supported. Up to 2 very minor phrasing additions that don't introduce new facts.",
        "score_3": "Most core claims are supported but 2-3 specific facts (names, numbers, dates) are not in the articles.",
        "score_2": "Many specific claims (numbers, names, events) cannot be found in the provided article text.",
        "score_1": "Answer introduces substantial new facts or directly contradicts the source articles.",
    },
    "relevance": {
        "score_5": "Answer directly and completely addresses exactly what was asked.",
        "score_4": "Answer mostly addresses the question with minor tangential content.",
        "score_3": "Answer partially addresses the question; some off-topic content.",
        "score_2": "Answer is mostly off-topic or misses the main point of the question.",
        "score_1": "Answer does not address the question at all.",
    },
    "completeness": {
        "score_5": "Every key fact present in the retrieved articles appears in the answer.",
        "score_4": "Nearly all key facts from the articles are covered; one minor detail missing.",
        "score_3": "At least half the key facts from the articles are covered.",
        "score_2": "Fewer than half the key facts from the articles are in the answer.",
        "score_1": "The answer covers almost none of the information in the retrieved articles.",
    },
    "conciseness": {
        "score_5": "Answer is focused and includes only necessary information.",
        "score_4": "Answer is slightly verbose but all content is relevant.",
        "score_3": "Answer contains noticeable padding or repeated information.",
        "score_2": "Answer is excessively long with significant irrelevant content.",
        "score_1": "Answer is padded to the point of being unusable.",
    },
}

TARGETS = {
    "faithfulness": 4.0,
    "relevance":    4.0,
    "completeness": 3.5,
    "conciseness":  3.5,
}


def judge_answer(query: str, answer: str, chunks: list[dict], metric: str) -> dict:
    """Call LLM judge to score one metric."""
    rubric = METRIC_RUBRICS[metric]
    # Give the judge more source context for a fair assessment
    chunks_text = "\n\n".join([
        f"[Article {i+1}] {c.get('source','Unknown')} ({(c.get('date',''))[:10]}):\n{c.get('text','')[:3000]}"
        for i, c in enumerate(chunks[:8])
    ])

    prompt = JUDGE_PROMPT.format(
        metric_upper=metric.upper(),
        query=query,
        answer=answer,
        chunks=chunks_text,
        **rubric,
    )

    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        max_completion_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()

    # Extract JSON from last line (after chain-of-thought reasoning)
    try:
        # Try last line first
        last_line = raw.strip().split("\n")[-1].strip()
        result = json.loads(last_line)
    except Exception:
        try:
            # Try finding any JSON object in the response
            import re
            match = re.search(r'\{[^{}]*"score"\s*:\s*\d[^{}]*\}', raw)
            result = json.loads(match.group()) if match else {"score": 3, "reason": "Parse error"}
        except Exception:
            result = {"score": 3, "reason": "Parse error"}

    # Clamp score to valid range
    result["score"] = max(1, min(5, int(result.get("score", 3))))
    return result


def run_answer_quality_evals(run_logs_dir: str = None) -> dict:
    """Score all logged runs with LLM-as-judge."""
    import glob

    if run_logs_dir is None:
        run_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    log_files = sorted(glob.glob(f"{run_logs_dir}/run_*.json"))
    if not log_files:
        print("[Evals] No run logs found. Run the agent first.")
        return {}

    # Deduplicate: keep most recent log per unique query
    query_to_log = {}
    for lf in log_files:
        with open(lf) as f:
            run = json.load(f)
        query_to_log[run["query"]] = run  # later files overwrite earlier
    unique_logs = list(query_to_log.values())

    print(f"[Evals] Scoring {len(unique_logs)} unique queries ({len(log_files)} total logs)")

    all_scores = {m: [] for m in METRICS}

    for run in unique_logs:
        query  = run["query"]
        answer = run["answer"]
        chunks = run.get("chunks_retrieved", [])
        if not isinstance(chunks, list):
            chunks = []

        if not answer or not answer.strip():
            print(f"  [Skip] Empty answer for: {query[:50]}")
            continue

        run_scores = {}
        for metric in METRICS:
            result = judge_answer(query, answer, chunks, metric)
            all_scores[metric].append(result["score"])
            run_scores[metric] = result["score"]

        print(f"  Q: {query[:55]}")
        print(f"     chunks={len(chunks)} | " +
              " | ".join(f"{m[:5]}={run_scores[m]}" for m in METRICS))

    if not any(all_scores.values()):
        print("[Evals] No scores computed.")
        return {}

    avg_scores = {
        m: round(sum(v) / len(v), 2)
        for m, v in all_scores.items() if v
    }

    print(f"\n{'='*45}")
    print("Layer 2 — Answer Quality Eval Results")
    print(f"{'='*45}")
    print(f"{'Metric':<16} {'Score':>8} {'Target':>8} {'Status':>10}")
    print("-" * 45)
    for metric, score in avg_scores.items():
        target = TARGETS[metric]
        status = "✅ Pass" if score >= target else "⚠️  Improve"
        print(f"{metric:<16} {score:>8.2f} {target:>8.1f} {status:>10}")

    return avg_scores


if __name__ == "__main__":
    run_answer_quality_evals()
