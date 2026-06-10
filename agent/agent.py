"""
Member B — Agentic Reasoning Loop
Agent dynamically decides which MCP tools to call based on what it finds
"""

import json
import openai
from tools import MCP_TOOLS, execute_tool
from dotenv import load_dotenv
import os

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a news research agent. You have access to these tools:
rewrite_query, search_news, filter_by_date, filter_by_source, fetch_full_article,
broaden_search, rerank_chunks, generate_summary, log_to_evals.

Rules:
1. Always call rewrite_query first to improve the search query.
2. Call search_news with the rewritten query.
3. If fewer than 5 chunks are returned, ALWAYS call broaden_search before continuing.
4. If the query mentions 'latest', 'today', or 'recent', call filter_by_date(days=7).
5. Always call rerank_chunks after retrieval (after broaden_search if used) to re-order by relevance.
6. Only call generate_summary when you have sufficient context (3+ chunks).
7. Never answer from memory. Only use retrieved chunks.
8. Always call log_to_evals after generating an answer.
9. For niche topics (space, quantum, robotics, satellites, autonomous vehicles), always call broaden_search even if search_news returns results.
"""

# Convert MCP tool schema (Anthropic format) to OpenAI function format
def to_openai_tools(mcp_tools: list) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
        }
        for t in mcp_tools
    ]

OPENAI_TOOLS = to_openai_tools(MCP_TOOLS)


def run_agent(user_query: str, verbose: bool = True) -> dict:
    """
    Main agent loop — runs until the agent stops calling tools.
    Returns: { answer, tool_trace, chunks_used }
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]
    tool_trace   = []
    final_chunks = []
    final_answer = ""

    if verbose:
        print(f"\n{'='*50}")
        print(f"[Agent] Query: {user_query}")
        print(f"{'='*50}")

    while True:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            max_completion_tokens=8096,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            messages=messages,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # No more tool calls — final answer
        if finish_reason == "stop" or not msg.tool_calls:
            final_answer = msg.content or ""
            if verbose:
                print(f"\n[Agent] Final answer:\n{final_answer}")
            break

        # Process tool calls
        messages.append(msg)  # add assistant message with tool_calls

        for tool_call in msg.tool_calls:
            tool_name   = tool_call.function.name
            tool_inputs = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"\n[Agent] → Calling: {tool_name}({json.dumps(tool_inputs)[:120]})")

            # Inject tool_trace into log_to_evals so Layer 4 evals work
            if tool_name == "log_to_evals":
                tool_inputs["tool_trace"] = tool_trace

            # Execute tool
            result = execute_tool(tool_name, tool_inputs)

            # Track
            tool_trace.append({
                "tool": tool_name,
                "inputs": tool_inputs,
                "result_preview": str(result)[:200],
            })

            if tool_name in ("search_news", "broaden_search", "filter_by_date",
                             "filter_by_source", "rerank_chunks"):
                if isinstance(result, list):
                    final_chunks = result

            # Feed result back
            result_str = json.dumps(result) if isinstance(result, (list, dict)) else str(result)
            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      result_str,
            })

    return {
        "query":       user_query,
        "answer":      final_answer,
        "tool_trace":  tool_trace,
        "chunks_used": final_chunks,
    }


if __name__ == "__main__":
    result = run_agent("What are the latest AI developments from OpenAI?")
    print(f"\nTool trace ({len(result['tool_trace'])} calls):")
    for step in result["tool_trace"]:
        print(f"  → {step['tool']}")
