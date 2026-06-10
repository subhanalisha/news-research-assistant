"""
Member C — Layer 4: Agent Behaviour Evals
Measures: Tool call accuracy, avg tool calls/query, broaden trigger rate, hallucination rate
"""

import json
import glob
import os


EXPECTED_TOOL_PATTERNS = {
    "standard": ["search_news", "generate_summary"],
    "recency": ["search_news", "filter_by_date", "generate_summary"],
    "thin_retrieval": ["search_news", "broaden_search", "generate_summary"],
}


def load_run_logs(run_logs_dir: str = None) -> list[dict]:
    if run_logs_dir is None:
        run_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    logs = []
    for f in glob.glob(f"{run_logs_dir}/run_*.json"):
        with open(f) as fh:
            logs.append(json.load(fh))
    return logs


def tool_call_accuracy(logs: list[dict], eval_dataset: list[dict]) -> float:
    """Check if agent called the right tools for each query type."""
    correct = 0
    for log, item in zip(logs, eval_dataset):
        expected_pattern = item.get("expected_pattern", "standard")
        expected_tools = EXPECTED_TOOL_PATTERNS.get(expected_pattern, [])
        actual_tools = [step["tool"] for step in log.get("tool_trace", [])]
        # Check all expected tools were called
        if all(t in actual_tools for t in expected_tools):
            correct += 1
    return correct / len(logs) if logs else 0.0


def avg_tool_calls(logs: list[dict]) -> float:
    """Mean number of tool calls per successful answer."""
    if not logs:
        return 0.0
    counts = [len(log.get("tool_trace", [])) for log in logs]
    return round(sum(counts) / len(counts), 2)


def broaden_trigger_rate(logs: list[dict], eval_dataset: list[dict]) -> float:
    """How often broaden_search was correctly triggered (when it should be)."""
    should_broaden = [item for item in eval_dataset if item.get("expected_pattern") == "thin_retrieval"]
    if not should_broaden:
        return 1.0
    triggered = 0
    for log in logs[:len(should_broaden)]:
        tools_used = [s["tool"] for s in log.get("tool_trace", [])]
        if "broaden_search" in tools_used:
            triggered += 1
    return triggered / len(should_broaden)


def hallucination_rate(logs: list[dict]) -> float:
    """Fraction of answers generated without calling search_news first."""
    bad = 0
    for log in logs:
        tools = [s["tool"] for s in log.get("tool_trace", [])]
        if "generate_summary" in tools:
            first_search = next((i for i, t in enumerate(tools) if t == "search_news"), None)
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
        "tool_call_accuracy": round(tool_call_accuracy(logs, eval_dataset), 3),
        "avg_tool_calls_per_query": avg_tool_calls(logs),
        "broaden_trigger_rate": round(broaden_trigger_rate(logs, eval_dataset), 3),
        "hallucination_rate": round(hallucination_rate(logs), 3),
    }

    targets = {
        "tool_call_accuracy": 0.85,
        "avg_tool_calls_per_query": None,   # target range: 2-4
        "broaden_trigger_rate": 0.80,
        "hallucination_rate": 0.0,
    }

    print(f"\n{'='*50}")
    print("Layer 4 — Agent Behaviour Eval Results")
    print(f"{'='*50}")
    print(f"{'Metric':<28} {'Score':>8} {'Target':>8} {'Status':>10}")
    print("-" * 50)
    for metric, score in results.items():
        target = targets[metric]
        if target is None:
            status = "✅ Pass" if 2 <= score <= 4 else "⚠️  Check"
            target_str = "2-4"
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
