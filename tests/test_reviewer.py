# ============================================================
# FILE: tests/test_reviewer.py
# ============================================================
# Tests for Phase 4 reviewer logic.
# Tests verdict parsing, reviewer agent creation, and review wiring.
# No LLM or Ollama needed — these test the parsing/wiring only.
#
# Usage: python -m tests.test_reviewer  (from project root)
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import parse_verdict, Orchestrator
from agents.reviewer import create_reviewer


# ---------------------------------------------------------------------------
# Verdict parsing tests
# ---------------------------------------------------------------------------

def test_verdict_pass():
    """Clean PASS verdict should parse correctly."""
    text = (
        "Thought: The code looks correct and runs without errors.\n"
        "Answer: The result is complete and accurate.\n\n"
        "VERDICT: PASS"
    )
    verdict, feedback = parse_verdict(text)
    assert verdict == "PASS"
    assert feedback is None
    print("  verdict PASS: PASSED")


def test_verdict_feedback():
    """FEEDBACK verdict with issues should parse correctly."""
    text = (
        "Thought: I found some problems.\n"
        "Answer: There are issues with the code.\n\n"
        "VERDICT: FEEDBACK\n"
        "- Issue 1: The function does not handle empty input\n"
        "- Issue 2: Missing error handling for file not found"
    )
    verdict, feedback = parse_verdict(text)
    assert verdict == "FEEDBACK"
    assert feedback is not None
    assert "empty input" in feedback
    assert "error handling" in feedback
    print("  verdict FEEDBACK: PASSED")


def test_verdict_feedback_single_issue():
    """FEEDBACK with one issue should still work."""
    text = "VERDICT: FEEDBACK\n- Issue 1: The result does not answer the original question"
    verdict, feedback = parse_verdict(text)
    assert verdict == "FEEDBACK"
    assert "original question" in feedback
    print("  verdict FEEDBACK single issue: PASSED")


def test_verdict_pass_case_insensitive():
    """Verdict parsing should be case-insensitive."""
    text = "verdict: pass"
    verdict, feedback = parse_verdict(text)
    assert verdict == "PASS"
    print("  verdict case insensitive: PASSED")


def test_verdict_with_think_tags():
    """DeepSeek-R1 may wrap in <think> tags — should still parse."""
    text = (
        "<think>Let me check this result carefully...</think>\n"
        "The code is well-written.\n"
        "VERDICT: PASS"
    )
    verdict, feedback = parse_verdict(text)
    assert verdict == "PASS"
    print("  verdict with think tags: PASSED")


def test_verdict_missing_defaults_to_pass():
    """If no VERDICT found, default to PASS to avoid blocking."""
    text = "The result looks fine to me. Everything checks out."
    verdict, feedback = parse_verdict(text)
    assert verdict == "PASS"
    assert feedback is None
    print("  verdict missing defaults to PASS: PASSED")


def test_verdict_feedback_no_details():
    """FEEDBACK with no detail text should still return FEEDBACK."""
    text = "VERDICT: FEEDBACK"
    verdict, feedback = parse_verdict(text)
    assert verdict == "FEEDBACK"
    assert feedback is not None  # should have fallback message
    assert "no details" in feedback.lower()
    print("  verdict FEEDBACK no details: PASSED")


def test_verdict_pass_beats_feedback_if_both():
    """If somehow both PASS and FEEDBACK appear, PASS wins (checked first)."""
    text = "VERDICT: PASS\nVERDICT: FEEDBACK\n- something wrong"
    verdict, feedback = parse_verdict(text)
    assert verdict == "PASS"
    print("  verdict PASS wins over FEEDBACK: PASSED")


# ---------------------------------------------------------------------------
# Reviewer agent creation tests
# ---------------------------------------------------------------------------

def test_create_reviewer():
    """Reviewer agent should have the right tools."""
    reviewer = create_reviewer(name="TestReviewer")
    assert reviewer.name == "TestReviewer"
    assert "read_file" in reviewer.tool_names
    assert "run_python_code" in reviewer.tool_names
    # Reviewer should NOT have write_file (it doesn't fix, only reviews)
    assert "write_file" not in reviewer.tool_names
    print("  reviewer creation: PASSED")


def test_orchestrator_creates_reviewer():
    """Orchestrator should create reviewer when agent_type is 'reviewer'."""
    orch = Orchestrator()
    reviewer = orch._create_agent("reviewer", 1)
    assert reviewer.name == "Reviewer_1"
    assert "read_file" in reviewer.tool_names
    assert "run_python_code" in reviewer.tool_names
    print("  orchestrator creates reviewer: PASSED")


# ---------------------------------------------------------------------------
# All Phase 3 tests still pass (kept from test_orchestrator.py)
# ---------------------------------------------------------------------------

def test_parse_plan_still_works():
    """Verify Phase 3 plan parsing is not broken."""
    from agents.orchestrator import parse_plan

    raw = '{"subtasks": [{"id": 1, "agent": "coder", "task": "Write code"}]}'
    result = parse_plan(raw)
    assert result is not None
    assert result[0]["agent"] == "coder"
    print("  plan parsing still works: PASSED")


if __name__ == "__main__":
    print("Running Phase 4 reviewer tests...\n")
    test_verdict_pass()
    test_verdict_feedback()
    test_verdict_feedback_single_issue()
    test_verdict_pass_case_insensitive()
    test_verdict_with_think_tags()
    test_verdict_missing_defaults_to_pass()
    test_verdict_feedback_no_details()
    test_verdict_pass_beats_feedback_if_both()
    test_create_reviewer()
    test_orchestrator_creates_reviewer()
    test_parse_plan_still_works()
    print("\nAll Phase 4 tests passed.")