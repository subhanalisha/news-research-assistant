"""
Helper script: run the agent against every query in eval_queries.json
to generate fresh run logs for Layer 2, 3, and 4 evals.

Usage:
    cd news-research-assistant
    python3 scripts/generate_eval_logs.py
"""
import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../agent"))
from agent import run_agent

EVAL_QUERIES_PATH = os.path.join(os.path.dirname(__file__), "../data/eval_queries.json")


def main():
    with open(EVAL_QUERIES_PATH) as f:
        eval_dataset = json.load(f)

    print(f"Running agent for {len(eval_dataset)} eval queries...\n")
    for i, item in enumerate(eval_dataset, 1):
        query = item["query"]
        print(f"\n[{i}/{len(eval_dataset)}] {query}")
        try:
            result = run_agent(query, verbose=False)
            print(f"  ✅ Done — {len(result['tool_trace'])} tool calls, "
                  f"{len(result['answer'])} char answer")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        time.sleep(1)   # small delay to avoid rate-limit

    print("\n✅ All eval logs generated. Now run evals:")
    print("   python3 evals/eval_retrieval.py")
    print("   python3 evals/eval_answer_quality.py")
    print("   python3 evals/eval_agent_behaviour.py")


if __name__ == "__main__":
    main()
