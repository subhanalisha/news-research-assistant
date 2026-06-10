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

JUDGE_PROMPT = """Score this answer on {metric_upper} (1-5):
5 = {score_5}
3 = {score_3}
1 = {score_1}

Question: {query}
Answer: {answer}
Articles: {chunks}

Return ONLY valid JSON: {{ "score": <1-5>, "reason": "<brief reason>" }}"""

METRIC_RUBRICS = {
    "faithfulness": {
        "score_5": "Every claim is directly supported by a source article.",
        "score_3": "Most claims supported; 1-2 unsupported additions.",
        "score_1": "Answer largely fabricated or contradicts sources.",
    },
    "relevance": {
        "score_5": "Answer directly and completely addresses what the user asked.",
        "score_3": "Answer mostly on-topic but includes tangential info.",
        "score_1": "Answer does not address the question.",
    },
    "completeness": {
        "score_5": "All key points from retrieved articles are reflected.",
        "score_3": "Most key points covered; some gaps.",
        "score_1": "Major key points missing.",
    },
    "conciseness": {
        "score_5": "Answer is appropriately brief without losing key info.",
        "score_3": "Slightly verbose but acceptable.",
        "score_1": "Excessively long or padded.",
    },
}

TARGETS = {
    "faithfulness": 4.0,
    "relevance": 4.0,
    "completeness": 3.5,
    "conciseness": 3.5,
}


def judge_answer(query: str, answer: str, chunks: list[dict], metric: str) -> dict:
    """Call LLM judge to score one metric."""
    rubric = METRIC_RUBRICS[metric]
    chunks_text = "\n".join([f"[{c.get('source','')}]: {c.get('text','')[:300]}" for c in chunks[:3]])

    prompt = JUDGE_PROMPT.format(
        metric_upper=metric.upper(),
        query=query,
        answer=answer,
        chunks=chunks_text,
        **rubric,
    )

    response = client.chat.completions.create(
        model="gpt-5.4-nano",   # cheaper model for eval scoring
        max_completion_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except Exception:
        result = {"score": 0, "reason": "Parse error"}
    return result


def run_answer_quality_evals(run_logs_dir: str = None) -> dict:
    """Score all logged runs with LLM-as-judge."""
    import glob
    from pathlib import Path

    if run_logs_dir is None:
        run_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    log_files = glob.glob(f"{run_logs_dir}/run_*.json")
    if not log_files:
        print("[Evals] No run logs found. Run the agent first.")
        return {}

    all_scores = {m: [] for m in METRICS}

    for log_file in log_files:
        with open(log_file) as f:
            run = json.load(f)

        query = run["query"]
        answer = run["answer"]
        chunks = run.get("chunks_retrieved", [])
        # Handle old log format where chunks_retrieved was saved as int
        if not isinstance(chunks, list):
            chunks = []

        for metric in METRICS:
            result = judge_answer(query, answer, chunks, metric)
            all_scores[metric].append(result["score"])

    # Aggregate
    avg_scores = {m: round(sum(v) / len(v), 2) for m, v in all_scores.items() if v}

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
