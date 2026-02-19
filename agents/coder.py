# ============================================================
# FILE: agents/coder.py
# ============================================================
# Phase 3: Coder Specialist Agent
#
# Specializes in writing, running, and debugging Python code.
# Uses a ReAct loop internally (inherits from core Agent).
#
# Tools: run_python_code, write_file, read_file
# ============================================================

from core.agent import Agent

CODER_PROMPT = """You are a Code Specialist. Your job is to write, run, and debug Python code.

When given a coding task:
1. Think about the approach before writing code
2. Write clean, working Python code
3. Run it to verify it works
4. If there are errors, read the error message, fix the code, and run again
5. If asked to save code to a file, use write_file

Always run your code at least once to verify it works before giving your final answer.
Include the working code AND the execution output in your Answer."""


def create_coder(name="Coder"):
    return Agent(
        name=name,
        system_prompt=CODER_PROMPT,
        tool_names=["run_python_code", "write_file", "read_file"],
    )