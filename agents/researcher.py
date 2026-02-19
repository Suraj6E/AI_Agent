# ============================================================
# FILE: agents/researcher.py
# ============================================================
# Phase 3: Researcher Specialist Agent
#
# Specializes in finding, reading, and summarizing information.
# Uses a ReAct loop internally (inherits from core Agent).
#
# Tools: web_search, read_file
# ============================================================

from core.agent import Agent

RESEARCHER_PROMPT = """You are a Research Specialist. Your job is to find and summarize information.

You are thorough and accurate. When given a research task:
1. Think about what information you need
2. Use your tools to find it
3. If one source is not enough, search again with different terms
4. Summarize your findings clearly with key facts

When you have gathered enough information, provide a clear summary as your Answer.
Do NOT make up information. If you cannot find something, say so."""


def create_researcher(name="Researcher"):
    return Agent(
        name=name,
        system_prompt=RESEARCHER_PROMPT,
        tool_names=["web_search", "read_file"],
    )