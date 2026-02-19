# ============================================================
# FILE: tests/test_orchestrator.py
# ============================================================
# Tests for Phase 3 orchestrator logic.
# Tests plan parsing, agent creation, and context building.
# No LLM or Ollama needed — these test the parsing/wiring only.
#
# Usage: python -m tests.test_orchestrator  (from project root)
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import parse_plan, Orchestrator


# ---------------------------------------------------------------------------
# Plan parsing tests
# ---------------------------------------------------------------------------

def test_parse_valid_plan():
    """Standard JSON plan should parse correctly."""
    raw = '{"subtasks": [{"id": 1, "agent": "researcher", "task": "Find info"}, {"id": 2, "agent": "coder", "task": "Write code"}]}'
    result = parse_plan(raw)
    assert result is not None
    assert len(result) == 2
    assert result[0]["agent"] == "researcher"
    assert result[1]["agent"] == "coder"
    print("  valid plan: PASSED")


def test_parse_single_subtask():
    """Plan with one subtask should work."""
    raw = '{"subtasks": [{"id": 1, "agent": "general", "task": "Answer directly"}]}'
    result = parse_plan(raw)
    assert result is not None
    assert len(result) == 1
    assert result[0]["agent"] == "general"
    print("  single subtask: PASSED")


def test_parse_plan_with_think_tags():
    """DeepSeek-R1 may wrap output in <think> tags before the JSON."""
    raw = (
        '<think>Let me analyze this task and decide how to split it...</think>\n'
        '{"subtasks": [{"id": 1, "agent": "researcher", "task": "Look up data"}]}'
    )
    result = parse_plan(raw)
    assert result is not None
    assert len(result) == 1
    assert result[0]["agent"] == "researcher"
    print("  plan with think tags: PASSED")


def test_parse_plan_with_surrounding_text():
    """Some models add text around the JSON. Parser should still find it."""
    raw = (
        'Here is my plan:\n'
        '{"subtasks": [{"id": 1, "agent": "coder", "task": "Write a script"}]}\n'
        'I hope this helps.'
    )
    result = parse_plan(raw)
    assert result is not None
    assert result[0]["agent"] == "coder"
    print("  plan with surrounding text: PASSED")


def test_parse_plan_invalid_json():
    """Malformed JSON should return None."""
    raw = '{"subtasks": [{"id": 1, "agent": "researcher" BROKEN'
    result = parse_plan(raw)
    assert result is None
    print("  invalid JSON: PASSED")


def test_parse_plan_no_subtasks_key():
    """JSON without a subtasks key should return None."""
    raw = '{"tasks": [{"id": 1, "agent": "researcher", "task": "Find info"}]}'
    result = parse_plan(raw)
    assert result is None
    print("  no subtasks key: PASSED")


def test_parse_plan_empty_subtasks():
    """Empty subtasks list should return None."""
    raw = '{"subtasks": []}'
    result = parse_plan(raw)
    assert result is None
    print("  empty subtasks: PASSED")


def test_parse_plan_no_json():
    """No JSON at all should return None."""
    raw = "I think we should research this topic and then write some code."
    result = parse_plan(raw)
    assert result is None
    print("  no JSON at all: PASSED")


# ---------------------------------------------------------------------------
# Agent creation tests
# ---------------------------------------------------------------------------

def test_create_agents():
    """Orchestrator should create the right agent type for each specialist."""
    orch = Orchestrator()

    researcher = orch._create_agent("researcher", 1)
    assert researcher.name == "Researcher_1"
    assert "web_search" in researcher.tool_names
    assert "read_file" in researcher.tool_names

    coder = orch._create_agent("coder", 2)
    assert coder.name == "Coder_2"
    assert "run_python_code" in coder.tool_names
    assert "write_file" in coder.tool_names

    general = orch._create_agent("general", 3)
    assert general.name == "General_3"
    assert "calculate" in general.tool_names

    # Unknown type should fall back to general
    fallback = orch._create_agent("unknown_type", 4)
    assert fallback.name == "Unknown_type_4"
    assert "calculate" in fallback.tool_names

    print("  agent creation: PASSED")


# ---------------------------------------------------------------------------
# Context building tests
# ---------------------------------------------------------------------------

def test_build_context():
    """Context from previous results should be formatted correctly."""
    orch = Orchestrator()

    results = [
        {"id": 1, "agent": "researcher", "task": "Find data", "result": "Found GDP is 25 trillion"},
        {"id": 2, "agent": "coder", "task": "Calculate", "result": "Growth rate is 2.5%"},
    ]

    context = orch._build_context(results)
    assert "Subtask 1" in context
    assert "researcher" in context
    assert "25 trillion" in context
    assert "Subtask 2" in context
    assert "2.5%" in context
    print("  context building: PASSED")


def test_build_context_truncation():
    """Long results should be truncated in context to keep prompts manageable."""
    orch = Orchestrator()

    long_result = "x" * 1000
    results = [{"id": 1, "agent": "researcher", "task": "Find data", "result": long_result}]

    context = orch._build_context(results)
    # _build_context truncates to 500 chars per result
    assert len(context) < 600
    print("  context truncation: PASSED")


if __name__ == "__main__":
    print("Running Phase 3 orchestrator tests...\n")
    test_parse_valid_plan()
    test_parse_single_subtask()
    test_parse_plan_with_think_tags()
    test_parse_plan_with_surrounding_text()
    test_parse_plan_invalid_json()
    test_parse_plan_no_subtasks_key()
    test_parse_plan_empty_subtasks()
    test_parse_plan_no_json()
    test_create_agents()
    test_build_context()
    test_build_context_truncation()
    print("\nAll Phase 3 tests passed.")