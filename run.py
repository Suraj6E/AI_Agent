# ============================================================
# FILE: run.py
# ============================================================
# The main entry point. Run this to talk to the agent system.
#
# Usage: python run.py
#
# Two modes:
#   1. single — one ReAct agent with all tools (Phase 2)
#   2. multi  — orchestrator + specialist agents (Phase 3)
#
# You can switch modes to compare results on the same task.
# Type 'mode' during a session to switch, 'quit' to exit.
# ============================================================

import os
from dotenv import load_dotenv
from core.agent import Agent
from core import llm_client
from agents.orchestrator import Orchestrator

load_dotenv()


SINGLE_AGENT_PROMPT = """You are a helpful assistant with access to tools.

When a user asks you something, decide if you need to use a tool or can answer directly.
Always think step by step before acting. Explain your reasoning clearly.
After gathering all information you need, give a clear, direct final answer."""


def run_single_agent(user_input, agent):
    """Run task through a single ReAct agent."""
    answer = agent.run(user_input)
    trace_path = agent.save_trace()
    print(f"\n  [Trace saved to {trace_path}]")
    return answer


def run_multi_agent(user_input, orchestrator):
    """Run task through the orchestrator + specialists."""
    answer = orchestrator.run(user_input)
    trace_path = orchestrator.save_trace()
    print(f"\n  [Trace saved to {trace_path}]")
    return answer


def main():
    print("=" * 60)
    print("Multi-Agent System — Phase 3: Orchestrator + Specialists")
    print("=" * 60)

    ok, status = llm_client.health_check()
    print(f"\n{status}")
    print(f"  Model:           {llm_client.MODEL_NAME}")
    print(f"  Ollama URL:      {llm_client.BASE_URL}")
    print(f"  LLM timeout:     {llm_client.TIMEOUT}s")
    print(f"  LLM retries:     {llm_client.MAX_RETRIES}")
    print(f"  Max tool rounds: {os.getenv('MAX_TOOL_ROUNDS', '10')}")
    print(f"  Verbose:         {os.getenv('VERBOSE', 'true')}")

    if not ok:
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama app is running (check system tray)")
        print("  2. If model is missing: ollama pull deepseek-r1:8b")
        print("  3. If Ollama not installed: https://ollama.com/download/windows")
        return

    # Create both runners
    single_agent = Agent(
        name="GeneralAgent",
        system_prompt=SINGLE_AGENT_PROMPT,
        tool_names=["calculate", "read_file", "write_file", "run_python_code", "web_search"],
    )
    orchestrator = Orchestrator()

    # Default to multi-agent mode (Phase 3)
    mode = "multi"

    print(f"\nMode: {mode}")
    print("  'mode'  — switch between single / multi")
    print("  'quit'  — exit")
    print("\nReady. Type your task.\n")

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

        if user_input.lower() == "mode":
            mode = "single" if mode == "multi" else "multi"
            print(f"\n  Switched to: {mode}\n")
            continue

        print()

        if mode == "single":
            answer = run_single_agent(user_input, single_agent)
        else:
            answer = run_multi_agent(user_input, orchestrator)

        print(f"\nAnswer: {answer}\n")
        print("-" * 40)


if __name__ == "__main__":
    main()