# ============================================================
# FILE: run.py
# ============================================================
# The main entry point. Run this to talk to the agent.
#
# Usage: python run.py
#
# What it does:
#   1. Checks if Ollama is running and model is available
#   2. Creates a ReAct agent with all tools
#   3. Loops: you type a task → agent reasons through it → prints answer
#   4. Type 'quit' to exit
#   5. Traces are saved to logs/agent_traces/ after each task
# ============================================================

import os
from dotenv import load_dotenv
from core.agent import Agent
from core import llm_client

load_dotenv()


SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

When a user asks you something, decide if you need to use a tool or can answer directly.
Always think step by step before acting. Explain your reasoning clearly.
After gathering all information you need, give a clear, direct final answer."""


def main():
    print("=" * 60)
    print("Multi-Agent System — Phase 2: ReAct Agent")
    print("  Thought → Act → Observe loop")
    print("=" * 60)

    ok, status = llm_client.health_check()
    print(f"\n{status}")
    print(f"  Model:           {llm_client.MODEL_NAME}")
    print(f"  Ollama URL:      {llm_client.BASE_URL}")
    print(f"  Max tool rounds: {os.getenv('MAX_TOOL_ROUNDS', '10')}")
    print(f"  Verbose:         {os.getenv('VERBOSE', 'true')}")

    if not ok:
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama app is running (check system tray)")
        print("  2. If model is missing: ollama pull deepseek-r1:8b")
        print("  3. If Ollama not installed: https://ollama.com/download/windows")
        return

    agent = Agent(
        name="GeneralAgent",
        system_prompt=SYSTEM_PROMPT,
        tool_names=["calculate", "read_file", "write_file", "run_python_code", "web_search"],
    )

    print("\nReady. Type your task (or 'quit' to exit).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        print()
        answer = agent.run(user_input)
        print(f"\nAnswer: {answer}\n")

        # Save the trace for this task so you can review the reasoning later
        trace_path = agent.save_trace()
        print(f"  [Trace saved to {trace_path}]")

        print("-" * 40)


if __name__ == "__main__":
    main()