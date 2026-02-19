# ============================================================
# FILE: core/agent.py
# ============================================================
# The agent loop with manual tool parsing.
#
# How it works:
#   1. System prompt tells the model what tools exist and what
#      format to use when calling them: [TOOL_CALL] {"name": ..., "arguments": ...}
#   2. Model outputs plain text. If it contains [TOOL_CALL], we parse it.
#   3. We execute the tool, add the result as [TOOL_RESULT], and loop.
#   4. If no [TOOL_CALL] found, the text is the final answer.
#
# This works with ANY model — no built-in tool support needed.
# ============================================================

import os
import json
import re
from dotenv import load_dotenv
from core import llm_client
from core.tools import execute_tool, get_tool_descriptions

load_dotenv()

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "10"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

TOOL_CALL_TAG = "[TOOL_CALL]"
TOOL_RESULT_TAG = "[TOOL_RESULT]"


def build_system_prompt(base_prompt, tool_names):
    """
    Build the full system prompt by appending tool descriptions
    and the format instructions for tool calling.
    """
    tool_desc = get_tool_descriptions(tool_names)

    return f"""{base_prompt}

You have access to the following tools:

{tool_desc}

IMPORTANT RULES FOR USING TOOLS:
1. To call a tool, write exactly this format on its own line:
   [TOOL_CALL] {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
2. Then STOP and wait. Do not write anything after [TOOL_CALL].
3. You will receive the result as [TOOL_RESULT] ...
4. After receiving a tool result, you can call another tool or give your final answer.
5. If you do NOT need a tool, just answer directly — no [TOOL_CALL] needed.
6. Only call ONE tool at a time.
7. The JSON after [TOOL_CALL] must be valid JSON on a single line."""


def parse_tool_call(text):
    """
    Look for [TOOL_CALL] in the model's output and parse the JSON after it.
    Returns (tool_name, arguments) if found, or (None, None) if not.
    """
    if TOOL_CALL_TAG not in text:
        return None, None

    idx = text.index(TOOL_CALL_TAG) + len(TOOL_CALL_TAG)
    remaining = text[idx:].strip()

    json_match = re.search(r'\{.*\}', remaining, re.DOTALL)
    if not json_match:
        return None, None

    try:
        parsed = json.loads(json_match.group())
        tool_name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        return tool_name, arguments
    except json.JSONDecodeError:
        return None, None


def extract_thinking(text):
    """
    DeepSeek-R1 wraps internal reasoning in <think>...</think> tags.
    Returns (thinking_text, remaining_text). If no thinking found, returns (None, original_text).
    """
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        remaining = text[:match.start()] + text[match.end():]
        return thinking, remaining.strip()
    return None, text


class Agent:
    def __init__(self, name, system_prompt, tool_names=None):
        self.name = name
        self.tool_names = tool_names or []
        self.system_prompt = build_system_prompt(system_prompt, self.tool_names) if self.tool_names else system_prompt
        self.history = []
        self.trace = []

    def run(self, user_input):
        self.history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        self.trace = []

        self._log(f"--- {self.name} started ---")
        self._log(f"Task: {user_input}")

        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            self._log(f"\n[Round {round_num}]")

            response_text = llm_client.chat(messages=self.history)

            if response_text.startswith("[ERROR]"):
                self._log(f"Error: {response_text}")
                return f"Agent error: {response_text}"

            self._log(f"Raw output:\n{response_text}\n")

            thinking, visible_text = extract_thinking(response_text)
            if thinking:
                self._log(f"[Thinking detected — {len(thinking)} chars]")

            tool_name, arguments = parse_tool_call(response_text)

            if tool_name is None:
                self._log(f"Final answer (no tool call found)")
                self.trace.append({"round": round_num, "type": "final_answer", "content": response_text})
                return response_text

            self._log(f"Tool call: {tool_name}({json.dumps(arguments)[:100]})")

            result = execute_tool(tool_name, arguments)

            self._log(f"Tool result: {result[:200]}")

            self.trace.append({
                "round": round_num,
                "type": "tool_call",
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            })

            self.history.append({"role": "assistant", "content": response_text})
            self.history.append({"role": "user", "content": f"{TOOL_RESULT_TAG} {result}"})

        self._log(f"Hit max tool rounds ({MAX_TOOL_ROUNDS}). Forcing final answer.")
        return self._force_final_answer()

    def _force_final_answer(self):
        self.history.append({
            "role": "user",
            "content": "You have used all available tool rounds. Give your best final answer now based on what you have. Do NOT call any more tools.",
        })

        response_text = llm_client.chat(messages=self.history)
        return response_text

    def get_trace(self):
        return self.trace

    def _log(self, message):
        if VERBOSE:
            print(f"[{self.name}] {message}")