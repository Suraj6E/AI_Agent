# ============================================================
# FILE: tests/test_react.py
# ============================================================
# Tests for Phase 2 ReAct parsing.
# These test the parsing logic only — no LLM or Ollama needed.
#
# Usage: python -m tests.test_react  (from project root)
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import (
    parse_tool_call,
    parse_react_thought,
    parse_react_answer,
    extract_thinking,
    build_system_prompt,
    clean_final_answer,
    _extract_first_json,
)


def test_parse_thought_with_act():
    """Model outputs Thought then Act — should extract the thought."""
    text = (
        'Thought: I need to calculate 2 + 3. I will use the calculate tool.\n'
        'Act: [TOOL_CALL] {"name": "calculate", "arguments": {"expression": "2 + 3"}}'
    )
    thought, _ = parse_react_thought(text)
    assert thought is not None
    assert "calculate 2 + 3" in thought
    print("  thought + act: PASSED")


def test_parse_thought_with_answer():
    """Model outputs Thought then Answer — should extract both."""
    text = (
        'Thought: The user asked a simple question. I know the answer.\n'
        'Answer: The capital of France is Paris.'
    )
    thought, _ = parse_react_thought(text)
    answer = parse_react_answer(text)
    assert thought is not None
    assert "simple question" in thought
    assert answer is not None
    assert "Paris" in answer
    print("  thought + answer: PASSED")


def test_parse_answer_multiline():
    """Answer section can span multiple lines."""
    text = (
        'Thought: I have all the data I need.\n'
        'Answer: Here are the results:\n'
        '- Item 1: 42\n'
        '- Item 2: 58\n'
        'Total: 100'
    )
    answer = parse_react_answer(text)
    assert answer is not None
    assert "100" in answer
    assert "Item 1" in answer
    print("  multiline answer: PASSED")


def test_parse_tool_call_in_act():
    """Tool call embedded in an Act line should still parse."""
    text = 'Act: [TOOL_CALL] {"name": "read_file", "arguments": {"file_path": "data.txt"}}'
    tool_name, args = parse_tool_call(text)
    assert tool_name == "read_file"
    assert args["file_path"] == "data.txt"
    print("  tool call in act: PASSED")


def test_no_thought():
    """If model skips the Thought tag, parser returns None."""
    text = 'I will just answer directly. The answer is 42.'
    thought, _ = parse_react_thought(text)
    assert thought is None
    print("  no thought tag: PASSED")


def test_no_answer_tag():
    """If model skips the Answer tag, parser returns None."""
    text = 'Thought: Thinking about it.\nThe answer is 42.'
    answer = parse_react_answer(text)
    assert answer is None
    print("  no answer tag: PASSED")


def test_deepseek_thinking_plus_react():
    """DeepSeek-R1 wraps internal reasoning in <think> tags before our ReAct format."""
    text = (
        '<think>Let me work through this problem step by step...</think>\n'
        'Thought: I need to calculate the sum.\n'
        'Act: [TOOL_CALL] {"name": "calculate", "arguments": {"expression": "10 + 20"}}'
    )
    thinking, visible = extract_thinking(text)
    assert thinking is not None
    assert "step by step" in thinking

    thought, _ = parse_react_thought(visible)
    assert thought is not None
    assert "calculate the sum" in thought

    tool_name, args = parse_tool_call(visible)
    assert tool_name == "calculate"
    assert args["expression"] == "10 + 20"
    print("  deepseek thinking + react: PASSED")


def test_system_prompt_contains_react_format():
    """System prompt should include Thought/Act/Answer instructions."""
    prompt = build_system_prompt("You are helpful.", ["calculate"])
    assert "Thought:" in prompt
    assert "Act:" in prompt
    assert "Answer:" in prompt
    assert "Observe:" in prompt
    assert "calculate" in prompt
    print("  system prompt format: PASSED")


def test_full_react_cycle_parsing():
    """Simulate a full Thought → Act → Observe → Thought → Answer cycle."""

    # Round 1: model reasons then calls a tool
    round1 = (
        'Thought: The user wants to know 15 * 7. I should use the calculate tool.\n'
        'Act: [TOOL_CALL] {"name": "calculate", "arguments": {"expression": "15 * 7"}}'
    )
    thought1, _ = parse_react_thought(round1)
    tool_name, args = parse_tool_call(round1)
    assert thought1 is not None
    assert tool_name == "calculate"
    assert args["expression"] == "15 * 7"

    # Round 2: model gets result and gives final answer
    round2 = (
        'Thought: The calculation returned 105. I have the answer.\n'
        'Answer: 15 × 7 = 105'
    )
    thought2, _ = parse_react_thought(round2)
    answer = parse_react_answer(round2)
    assert thought2 is not None
    assert "105" in thought2
    assert answer is not None
    assert "105" in answer
    print("  full react cycle: PASSED")


# ---------------------------------------------------------------------------
# New tests for _extract_first_json (balanced brace extraction)
# ---------------------------------------------------------------------------

def test_extract_first_json_simple():
    """Simple JSON object should extract correctly."""
    text = '{"name": "calculate", "arguments": {"expression": "2+3"}}'
    result = _extract_first_json(text)
    assert result is not None
    assert '"calculate"' in result
    print("  extract first json simple: PASSED")


def test_extract_first_json_with_surrounding_text():
    """JSON embedded in other text should still extract."""
    text = 'some prefix {"name": "test", "arguments": {}} some suffix'
    result = _extract_first_json(text)
    assert result == '{"name": "test", "arguments": {}}'
    print("  extract first json with text: PASSED")


def test_extract_first_json_nested():
    """Nested JSON should extract the complete outer object."""
    text = '{"name": "write_file", "arguments": {"path": "a.txt", "content": "hello"}}'
    result = _extract_first_json(text)
    assert result is not None
    import json
    parsed = json.loads(result)
    assert parsed["name"] == "write_file"
    assert parsed["arguments"]["content"] == "hello"
    print("  extract first json nested: PASSED")


def test_extract_first_json_with_string_braces():
    """Braces inside JSON strings should not break the extraction."""
    text = '{"name": "test", "arguments": {"code": "if x > 0 { return }"}}'
    result = _extract_first_json(text)
    assert result is not None
    import json
    parsed = json.loads(result)
    assert "{" in parsed["arguments"]["code"]
    print("  extract first json with string braces: PASSED")


def test_extract_first_json_stops_at_first():
    """When multiple JSON objects exist, only the first is extracted."""
    text = (
        '{"name": "write_file", "arguments": {"path": "a.txt", "content": "hi"}}\n'
        'Observe: done\n'
        '{"name": "read_file", "arguments": {"file_path": "a.txt"}}'
    )
    result = _extract_first_json(text)
    assert result is not None
    import json
    parsed = json.loads(result)
    assert parsed["name"] == "write_file"
    print("  extract first json stops at first: PASSED")


def test_extract_first_json_no_json():
    """No JSON at all should return None."""
    result = _extract_first_json("just plain text here")
    assert result is None
    print("  extract first json no json: PASSED")


# ---------------------------------------------------------------------------
# New tests for multi-tool-call hallucination (the actual bug)
# ---------------------------------------------------------------------------

def test_parse_tool_call_multi_hallucination():
    """
    Model hallucinates a full conversation with two tool calls.
    parse_tool_call should extract only the FIRST tool call successfully.
    This was the bug: greedy regex matched across both, creating invalid JSON.
    """
    text = (
        'Thought: I need to write the essay and create an image.\n'
        'Act: [TOOL_CALL] {"name": "write_file", "arguments": {"file_path": "essay.txt", "content": "Cats are great."}}\n'
        '\n'
        'Observe: Successfully wrote 15 characters to essay.txt\n'
        '\n'
        'Thought: Now create the image.\n'
        'Act: [TOOL_CALL] {"name": "write_file", "arguments": {"file_path": "cat.txt", "content": "meow"}}\n'
        '\n'
        'Observe: Successfully wrote 4 characters to cat.txt\n'
        '\n'
        'Answer: Done!'
    )
    tool_name, args = parse_tool_call(text)
    assert tool_name == "write_file", f"Expected write_file, got {tool_name}"
    assert args["file_path"] == "essay.txt", f"Expected essay.txt, got {args.get('file_path')}"
    assert args["content"] == "Cats are great."
    print("  multi-tool-call hallucination: PASSED")


# ---------------------------------------------------------------------------
# New tests for clean_final_answer
# ---------------------------------------------------------------------------

def test_clean_final_answer_with_answer_tag():
    """If Answer: tag exists, extract just that."""
    text = (
        'Thought: I am done.\n'
        'Answer: The result is 42.'
    )
    result = clean_final_answer(text)
    assert result == "The result is 42."
    print("  clean answer with tag: PASSED")


def test_clean_final_answer_hallucinated_conversation():
    """Strip a full hallucinated Thought/Act/Observe conversation."""
    text = (
        'Thought: I need to write a file.\n'
        'Act: [TOOL_CALL] {"name": "write_file", "arguments": {"file_path": "a.txt", "content": "hello"}}\n'
        '\n'
        'Observe: Successfully wrote 5 characters to a.txt\n'
        '\n'
        'Thought: Now I am done.\n'
        'Answer: I wrote the file for you.'
    )
    result = clean_final_answer(text)
    assert result == "I wrote the file for you."
    assert "TOOL_CALL" not in result
    assert "Observe:" not in result
    print("  clean hallucinated conversation: PASSED")


def test_clean_final_answer_thought_only():
    """If only Thought: lines with no Answer:, extract the thought content."""
    text = 'Thought: The answer is simply 42.'
    result = clean_final_answer(text)
    assert "42" in result
    assert not result.startswith("Thought:")
    print("  clean thought only: PASSED")


def test_clean_final_answer_plain_text():
    """Plain text without any tags should pass through unchanged."""
    text = "The capital of France is Paris."
    result = clean_final_answer(text)
    assert result == text
    print("  clean plain text: PASSED")


if __name__ == "__main__":
    print("Running Phase 2 ReAct parsing tests...\n")
    test_parse_thought_with_act()
    test_parse_thought_with_answer()
    test_parse_answer_multiline()
    test_parse_tool_call_in_act()
    test_no_thought()
    test_no_answer_tag()
    test_deepseek_thinking_plus_react()
    test_system_prompt_contains_react_format()
    test_full_react_cycle_parsing()

    print("\n--- JSON extraction tests ---\n")
    test_extract_first_json_simple()
    test_extract_first_json_with_surrounding_text()
    test_extract_first_json_nested()
    test_extract_first_json_with_string_braces()
    test_extract_first_json_stops_at_first()
    test_extract_first_json_no_json()

    print("\n--- Bug fix tests ---\n")
    test_parse_tool_call_multi_hallucination()

    print("\n--- clean_final_answer tests ---\n")
    test_clean_final_answer_with_answer_tag()
    test_clean_final_answer_hallucinated_conversation()
    test_clean_final_answer_thought_only()
    test_clean_final_answer_plain_text()

    print("\nAll Phase 2 tests passed.")