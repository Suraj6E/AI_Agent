# ============================================================
# FILE: core/agent.py
# ============================================================
# Phase 2: ReAct Agent Loop (Thought → Act → Observe)
#
# Changes from Phase 1:
#   - System prompt now instructs the model to always output a
#     Thought before acting or answering.
#   - Agent loop parses Thought, Act, and Observe steps.
#   - Trace stores each reasoning step for debugging.
#   - Colored/labeled console output shows the ReAct cycle.
#
# The model must follow this format:
#   Thought: <reasoning about what to do next>
#   Act: [TOOL_CALL] {"name": "...", "arguments": {...}}
#     — OR —
#   Thought: <reasoning>
#   Answer: <final answer to the user>
#
# After a tool runs, the model receives:
#   Observe: <tool result>
#
# This works with ANY model — no built-in tool support needed.
# ============================================================

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from core import llm_client
from core.tools import execute_tool, get_tool_descriptions

load_dotenv()

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "10"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

TOOL_CALL_TAG = "[TOOL_CALL]"
TOOL_RESULT_TAG = "[TOOL_RESULT]"


# ---------------------------------------------------------------------------
# System prompt builder — ReAct format
# ---------------------------------------------------------------------------

def build_system_prompt(base_prompt, tool_names):
    tool_desc = get_tool_descriptions(tool_names)

    return f"""{base_prompt}

You have access to the following tools:

{tool_desc}

YOU MUST ALWAYS FOLLOW THIS FORMAT:

1. ALWAYS start with a Thought — reason about what you know and what you need.
2. Then EITHER use a tool OR give your final answer.

FORMAT WHEN USING A TOOL:
Thought: <your reasoning about what to do next>
Act: [TOOL_CALL] {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}

FORMAT WHEN GIVING YOUR FINAL ANSWER (no more tools needed):
Thought: <your reasoning about why you are done>
Answer: <your final answer to the user>

RULES:
- Every response MUST start with "Thought:"
- After Act, STOP. Do not write anything after [TOOL_CALL]. Wait for the result.
- You will receive tool results as: Observe: <result>
- After Observe, start your next Thought.
- Only call ONE tool at a time.
- The JSON after [TOOL_CALL] must be valid JSON on a single line.
- If you do NOT need any tools, go straight to Thought + Answer.
- Always end with "Answer:" when you have your final response."""


# ---------------------------------------------------------------------------
# The original Phase 1 system prompt builder is below.
# It can be removed now that ReAct is in place, but is kept
# in case you want to compare Phase 1 vs Phase 2 behavior.
# ---------------------------------------------------------------------------

def build_system_prompt_phase1(base_prompt, tool_names):
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


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_tool_call(text):
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
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        remaining = text[:match.start()] + text[match.end():]
        return thinking, remaining.strip()
    return None, text


def parse_react_thought(text):
    """
    Extract the Thought: line(s) from the model output.
    Returns (thought_text, remaining_text).
    If no Thought: found, returns (None, original_text).
    """
    match = re.search(r'Thought:\s*(.*?)(?=\nAct:|\nAnswer:|\Z)', text, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        return thought, text
    return None, text


def parse_react_answer(text):
    """
    Extract the Answer: section from the model output.
    Returns the answer text if found, None otherwise.
    """
    match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Agent class — ReAct loop
# ---------------------------------------------------------------------------

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
            self._log(f"\n{'='*40}")
            self._log(f"[Round {round_num}]")

            response_text = llm_client.chat(messages=self.history)

            if response_text.startswith("[ERROR]"):
                self._log(f"Error: {response_text}")
                return f"Agent error: {response_text}"

            # DeepSeek-R1 internal thinking (separate from our ReAct Thought)
            thinking, visible_text = extract_thinking(response_text)
            if thinking:
                self._log(f"  [Internal thinking — {len(thinking)} chars]")

            # Parse the ReAct Thought step
            thought, _ = parse_react_thought(visible_text)
            if thought:
                self._log(f"  Thought: {thought}")
            else:
                self._log(f"  (no Thought: tag found — model may not be following format)")

            # Check if model gave a final Answer
            answer = parse_react_answer(visible_text)
            if answer and TOOL_CALL_TAG not in visible_text:
                self._log(f"  Answer: {answer[:200]}")
                self.trace.append({
                    "round": round_num,
                    "type": "final_answer",
                    "thought": thought,
                    "answer": answer,
                    "raw": response_text,
                })
                return answer

            # Check if model is calling a tool (Act step)
            tool_name, arguments = parse_tool_call(response_text)

            if tool_name is None:
                # No tool call and no Answer: tag — treat entire response as final answer
                self._log(f"  (no tool call, no Answer: tag — treating as final answer)")
                self.trace.append({
                    "round": round_num,
                    "type": "final_answer",
                    "thought": thought,
                    "answer": visible_text,
                    "raw": response_text,
                })
                return visible_text

            # Tool call found — execute it
            self._log(f"  Act: {tool_name}({json.dumps(arguments)[:100]})")

            result = execute_tool(tool_name, arguments)

            self._log(f"  Observe: {result[:200]}")

            self.trace.append({
                "round": round_num,
                "type": "tool_call",
                "thought": thought,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "raw": response_text,
            })

            # Feed back into conversation using Observe: format
            self.history.append({"role": "assistant", "content": response_text})
            self.history.append({"role": "user", "content": f"Observe: {result}"})

        self._log(f"  Hit max tool rounds ({MAX_TOOL_ROUNDS}). Forcing final answer.")
        return self._force_final_answer()

    def _force_final_answer(self):
        self.history.append({
            "role": "user",
            "content": (
                "You have used all available tool rounds. "
                "Give your best final answer now based on what you have so far.\n\n"
                "Thought: <summarize what you know>\n"
                "Answer: <your final answer>"
            ),
        })

        response_text = llm_client.chat(messages=self.history)

        answer = parse_react_answer(response_text)
        if answer:
            return answer
        return response_text

    def get_trace(self):
        return self.trace

    def save_trace(self, directory="logs/agent_traces"):
        """
        Save the full trace to a JSON file for later analysis.
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(directory, f"{self.name}_{timestamp}.json")

        trace_data = {
            "agent": self.name,
            "timestamp": timestamp,
            "rounds": len(self.trace),
            "trace": self.trace,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)

        return filepath

    def _log(self, message):
        if VERBOSE:
            print(f"[{self.name}] {message}")