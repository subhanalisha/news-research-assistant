"""
Member C — Layer 4: Agent Behaviour Evals
Measures: Tool call accuracy, avg tool calls/query, broaden trigger rate, hallucination rate
"""

import json
import glob
import os
from difflib import get_close_matches


# Expected tool patterns — must include rewrite_query and rerank_chunks (added v2)
EXPECTED_TOOL_PATTERNS = {
    "standard":      ["rewrite_query", "search_news", "rerank_chunks", "generate_summary"],
    "recency":       ["rewrite_query", "search_news", "filter_by_date", "rerank_chunks", "generate_summary"],
    "thin_retrieval":["rewrite_query", "search_news", "broaden_search", "rerank_chunks", "generate_summary"],
}


def load_run_logs(run_logs_dir: str = None) -> list[dict]:
    if run_logs_dir is None:
        run_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    logs = []
    for f in sorted(glob.glob(f"{run_logs_dir}/run_*.json")):
        with open(f) as fh:
            logs.append(json.load(fh))
    return logs


def build_query_log_map(logs: list[dict]) -> dict:
    """
    Build a dict: query_text → best (most recent) log for that query.
    Sorted logs are in timestamp order so later entries overwrite earlier ones,
    giving us the most recent run per unique query.
    """
    query_map = {}
    for log in logs:
        query_map[log["query"]] = log
    return query_map


def find_log_for_query(eval_query: str, query_map: dict):
    """
    Find the best matching log for an eval query using:
    1. Exact match
    2. Fuzzy match (difflib, cutoff=0.55) — handles slight wording differences
    """
    if eval_query in query_map:
        return query_map[eval_query]
    matches = get_close_matches(eval_query, query_map.keys(), n=1, cutoff=0.55)
    if matches:
        return query_map[matches[0]]
    # Last resort: substring match (eval query words in log query)
    eval_words = set(eval_query.lower().split())
    best_overlap, best_log = 0, None
    for log_query, log in query_map.items():
        log_words = set(log_query.lower().split())
        overlap = len(eval_words & log_words) / max(len(eval_words), 1)
        if overlap > best_overlap:
            best_overlap, best_log = overlap, log
    return best_log if best_overlap >= 0.5 else None


def tool_call_accuracy(eval_dataset: list[dict], query_map: dict) -> tuple[float, list]:
    """
    Check if agent called the right tools for each query type.
    Matches logs to eval queries by query TEXT, not by list index.
    """
    correct = 0
    total = 0
    details = []
    for item in eval_dataset:
        log = find_log_for_query(item["query"], query_map)
        if log is None:
            details.append({"query": item["query"], "status": "no_log"})
            continue
        expected_pattern = item.get("expected_pattern", "standard")
        expected_tools = EXPECTED_TOOL_PATTERNS.get(expected_pattern, [])
        actual_tools = [step["tool"] for step in log.get("tool_trace", [])]
        passed = all(t in actual_tools for t in expected_tools)
        if passed:
            correct += 1
        total += 1
        details.append({
            "query": item["query"][:50],
            "pattern": expected_pattern,
            "expected": expected_tools,
            "actual": actual_tools,
            "passed": passed,
        })
    return (correct / total if total else 0.0), details


def avg_tool_calls(logs: list[dict]) -> float:
    """Mean number of tool calls per successful answer."""
    if not logs:
        return 0.0
    counts = [len(log.get("tool_trace", [])) for log in logs]
    return round(sum(counts) / len(counts), 2)


def broaden_trigger_rate(eval_dataset: list[dict], query_map: dict) -> tuple[float, list]:
    """
    How often broaden_search was correctly triggered for thin_retrieval queries.
    Matches by query TEXT, not by index.
    """
    thin_items = [item for item in eval_dataset
                  if item.get("expected_pattern") == "thin_retrieval"]
    if not thin_items:
        return 1.0, []
    triggered = 0
    details = []
    for item in thin_items:
        log = find_log_for_query(item["query"], query_map)
        if log is None:
            details.append({"query": item["query"][:50], "triggered": False, "reason": "no_log"})
            continue
        tools_used = [s["tool"] for s in log.get("tool_trace", [])]
        did_broaden = "broaden_search" in tools_used
        if did_broaden:
            triggered += 1
        details.append({
            "query": item["query"][:55],
            "triggered": did_broaden,
            "tools": tools_used,
        })
    return triggered / len(thin_items), details


def hallucination_rate(logs: list[dict]) -> float:
    """Fraction of answers generated without calling search_news first."""
    bad = 0
    for log in logs:
        tools = [s["tool"] for s in log.get("tool_trace", [])]
        if "generate_summary" in tools:
            first_search  = next((i for i, t in enumerate(tools) if t == "search_news"), None)
            first_summary = next((i for i, t in enumerate(tools) if t == "generate_summary"), None)
            if first_search is None or (first_summary is not None and first_summary < first_search):
                bad += 1
    return bad / len(logs) if logs else 0.0


def run_agent_behaviour_evals(eval_dataset_path: str = None) -> dict:
    """Run all agent behaviour evals."""
    if eval_dataset_path is None:
        eval_dataset_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../data/eval_queries.json"
        )
    with open(eval_dataset_path) as f:
        eval_dataset = json.load(f)

    logs = load_run_logs()
    if not logs:
        print("[Evals] No run logs found. Run the agent first.")
        return {}

    # Build query → best log mapping (text-based, not index-based)
    query_map = build_query_log_map(logs)
    print(f"[Evals] {len(logs)} total logs → {len(query_map)} unique queries")

    tca_score, tca_details = tool_call_accuracy(eval_dataset, query_map)
    btr_score, btr_details = broaden_trigger_rate(eval_dataset, query_map)

    results = {
        "tool_call_accuracy":       round(tca_score, 3),
        "avg_tool_calls_per_query": avg_tool_calls(logs),
        "broaden_trigger_rate":     round(btr_score, 3),
        "hallucination_rate":       round(hallucination_rate(logs), 3),
    }

    targets = {
        "tool_call_accuracy":       0.85,
        "avg_tool_calls_per_query": None,   # range 4-6
        "broaden_trigger_rate":     0.80,
        "hallucination_rate":       0.0,
    }

    print(f"\n{'='*55}")
    print("Layer 4 — Agent Behaviour Eval Results")
    print(f"{'='*55}")
    print(f"{'Metric':<28} {'Score':>8} {'Target':>8} {'Status':>10}")
    print("-" * 55)
    for metric, score in results.items():
        target = targets[metric]
        if target is None:
            status = "✅ Pass" if 4 <= score <= 6 else "⚠️  Check"
            target_str = "4-6"
        elif metric == "hallucination_rate":
            status = "✅ Pass" if score == 0.0 else "❌ Fail"
            target_str = "0%"
        else:
            status = "✅ Pass" if score >= target else "⚠️  Improve"
            target_str = f"{target:.0%}"
        print(f"{metric:<28} {score:>8.3f} {target_str:>8} {status:>10}")

    # ── Verbose breakdown for failing metrics ───────────────────────────────
    print(f"\n── tool_call_accuracy breakdown ──")
    for d in tca_details:
        icon = "✅" if d.get("passed") else ("⏭" if d.get("status") == "no_log" else "❌")
        print(f"  {icon} [{d.get('pattern','?'):14}] {d['query'][:52]}")

    print(f"\n── broaden_trigger_rate breakdown (thin_retrieval only) ──")
    for d in btr_details:
        icon = "✅" if d.get("triggered") else "❌"
        print(f"  {icon} {d['query']}")
        print(f"       tools: {d.get('tools', [])}")

    return results


if __name__ == "__main__":
    run_agent_behaviour_evals()
