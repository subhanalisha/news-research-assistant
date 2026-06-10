"""
Member B — Agentic Reasoning Loop
Agent dynamically decides which MCP tools to call based on what it finds
"""

import anthropic
import json
from tools import MCP_TOOLS, execute_tool
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a news research agent. You have access to these tools:
search_news, filter_by_date, filter_by_source, fetch_full_article,
broaden_search, rerank_chunks, generate_summary, log_to_evals.

Rules:
1. Always call search_news first.
2. If fewer than 3 chunks are returned, call broaden_search.
3. If the query mentions 'latest', 'today', or 'recent', call filter_by_date(days=3).
4. Only call generate_summary when you have sufficient context (3+ chunks).
5. Never answer from memory. Only use retrieved chunks.
6. Always call log_to_evals after generating an answer.
"""


def run_agent(user_query: str, verbose: bool = True) -> dict:
    """
    Main agent loop — runs until the agent reaches end_turn.
    Returns: { answer, tool_trace, chunks_used }
    """
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_query}]
    tool_trace = []
    final_chunks = []
    final_answer = ""

    if verbose:
        print(f"\n{'='*50}")
        print(f"[Agent] Query: {user_query}")
        print(f"{'='*50}")

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            tools=MCP_TOOLS,
            messages=messages,
        )

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Extract final text answer
            for block in response.content:
                if hasattr(block, "text"):
                    final_answer = block.text
            if verbose:
                print(f"\n[Agent] Final answer:\n{final_answer}")
            break

        if response.stop_reason == "tool_use":
            # Process all tool calls in this response
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_inputs = block.input

                if verbose:
                    print(f"\n[Agent] → Calling: {tool_name}({json.dumps(tool_inputs)[:120]})")

                # Execute the tool
                result = execute_tool(tool_name, tool_inputs)

                # Track tool call
                tool_trace.append({
                    "tool": tool_name,
                    "inputs": tool_inputs,
                    "result_preview": str(result)[:200],
                })

                # Track chunks for eval logging
                if tool_name in ("search_news", "broaden_search", "filter_by_date",
                                 "filter_by_source", "rerank_chunks"):
                    if isinstance(result, list):
                        final_chunks = result

                # Serialize result for message
                if isinstance(result, (list, dict)):
                    result_str = json.dumps(result)
                else:
                    result_str = str(result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            # Feed tool results back to agent
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason
            print(f"[Agent] Unexpected stop reason: {response.stop_reason}")
            break

    return {
        "query": user_query,
        "answer": final_answer,
        "tool_trace": tool_trace,
        "chunks_used": final_chunks,
    }


if __name__ == "__main__":
    result = run_agent("What are the latest AI developments from OpenAI?")
    print(f"\nTool trace ({len(result['tool_trace'])} calls):")
    for step in result["tool_trace"]:
        print(f"  → {step['tool']}")
