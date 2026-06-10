"""
Helper script: run the agent against every query in eval_queries.json
to generate fresh run logs for Layer 2, 3, and 4 evals.

Usage:
    cd news-research-assistant
    python3 scripts/generate_eval_logs.py [--clean]

Flags:
    --clean   Delete all existing run logs before generating new ones
              (avoids stale index-mismatch issues in evals)
"""
import json
import os
import sys
import time
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../agent"))
from agent import run_agent

EVAL_QUERIES_PATH = os.path.join(os.path.dirname(__file__), "../data/eval_queries.json")
RESULTS_DIR       = os.path.join(os.path.dirname(__file__), "../evals/results")


def clean_logs():
    """Remove all existing run logs."""
    files = glob.glob(os.path.join(RESULTS_DIR, "run_*.json"))
    for f in files:
        os.remove(f)
    print(f"[Cleanup] Removed {len(files)} old run logs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true",
                        help="Delete old run logs before generating new ones")
    args = parser.parse_args()

    if args.clean:
        clean_logs()

    with open(EVAL_QUERIES_PATH) as f:
        eval_dataset = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Running agent for {len(eval_dataset)} eval queries...\n")
    passed = 0
    failed = 0

    for i, item in enumerate(eval_dataset, 1):
        query   = item["query"]
        pattern = item.get("expected_pattern", "standard")
        print(f"\n[{i}/{len(eval_dataset)}] [{pattern:14}] {query}")
        try:
            result = run_agent(query, verbose=False)
            tools  = [s["tool"] for s in result["tool_trace"]]
            print(f"  ✅ {len(result['tool_trace'])} tool calls: {tools}")
            print(f"     Answer: {result['answer'][:80]}...")
            passed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
        time.sleep(1)   # small delay to avoid rate-limit

    print(f"\n{'='*50}")
    print(f"Done — {passed} passed, {failed} failed")
    print(f"\nNow run evals:")
    print("   python3 evals/eval_retrieval.py")
    print("   python3 evals/eval_answer_quality.py")
    print("   python3 evals/eval_agent_behaviour.py")


if __name__ == "__main__":
    main()
