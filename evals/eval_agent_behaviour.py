"""
Member C — Layer 4: Agent Behaviour Evals
Measures: Tool call accuracy, avg tool calls/query, broaden trigger rate, hallucination rate
"""

import json
import glob
import os


# Updated patterns to include rewrite_query and rerank_chunks (added in v2)
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


def tool_call_accuracy(logs: list[dict], eval_dataset: list[dict]) -> float:
    """Check if agent called the right tools for each query type."""
    correct = 0
    total = min(len(logs), len(eval_dataset))
    for i in range(total):
        log  = logs[i]
        item = eval_dataset[i]
        expected_pattern = item.get("expected_pattern", "standard")
        expected_tools   = EXPECTED_TOOL_PATTERNS.get(expected_pattern, [])
        actual_tools     = [step["tool"] for step in log.get("tool_trace", [])]
        if all(t in actual_tools for t in expected_tools):
            correct += 1
    return correct / total if total else 0.0


def avg_tool_calls(logs: list[dict]) -> float:
    """Mean number of tool calls per successful answer."""
    if not logs:
        return 0.0
    counts = [len(log.get("tool_trace", [])) for log in logs]
    return round(sum(counts) / len(counts), 2)


def broaden_trigger_rate(logs: list[dict], eval_dataset: list[dict]) -> float:
    """How often broaden_search was correctly triggered for thin_retrieval queries."""
    thin_indices = [i for i, item in enumerate(eval_dataset)
                    if item.get("expected_pattern") == "thin_retrieval"]
    if not thin_indices:
        return 1.0
    triggered = 0
    for i in thin_indices:
        if i >= len(logs):
            continue
        tools_used = [s["tool"] for s in logs[i].get("tool_trace", [])]
        if "broaden_search" in tools_used:
            triggered += 1
    return triggered / len(thin_indices)


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
        eval_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/eval_queries.json")
    with open(eval_dataset_path) as f:
        eval_dataset = json.load(f)

    logs = load_run_logs()
    if not logs:
        print("[Evals] No run logs found. Run the agent first.")
        return {}

    results = {
        "tool_call_accuracy":      round(tool_call_accuracy(logs, eval_dataset), 3),
        "avg_tool_calls_per_query": avg_tool_calls(logs),
        "broaden_trigger_rate":    round(broaden_trigger_rate(logs, eval_dataset), 3),
        "hallucination_rate":      round(hallucination_rate(logs), 3),
    }

    # Updated targets — avg_tool_calls is now 4-6 with rewrite_query + rerank_chunks
    targets = {
        "tool_call_accuracy":      0.85,
        "avg_tool_calls_per_query": None,   # range: 4-6
        "broaden_trigger_rate":    0.80,
        "hallucination_rate":      0.0,
    }

    print(f"\n{'='*50}")
    print("Layer 4 — Agent Behaviour Eval Results")
    print(f"{'='*50}")
    print(f"{'Metric':<28} {'Score':>8} {'Target':>8} {'Status':>10}")
    print("-" * 50)
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

    return results


if __name__ == "__main__":
    run_agent_behaviour_evals()
