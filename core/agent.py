# ============================================================
# FILE: core/agent.py
# ============================================================
# The agent loop:
#   1. Send user task + tool schemas to LLM
#   2. If LLM returns tool calls → execute them → feed results back
#   3. If LLM returns text → that's the final answer
#   4. Repeat until done or max rounds hit
#
# Phase 2 will add ReAct (Thought/Act/Observe) on top of this loop.
# ============================================================

import os
import json
from dotenv import load_dotenv
from core import llm_client
from core.tools import execute_tool, get_tool_schemas

load_dotenv()

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "10"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"


class Agent:
    def __init__(self, name, system_prompt, tool_names=None):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_names = tool_names
        self.tool_schemas = get_tool_schemas(tool_names) if tool_names else []
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

            api_result = llm_client.chat(
                messages=self.history,
                tools=self.tool_schemas if self.tool_schemas else None,
            )

            parsed = llm_client.extract_response(api_result)

            if parsed["type"] == "error":
                self._log(f"Error: {parsed['content']}")
                return f"Agent error: {parsed['content']}"

            if parsed["type"] == "text":
                self._log(f"Final answer: {parsed['content'][:200]}...")
                self.trace.append({"round": round_num, "type": "final_answer", "content": parsed["content"]})
                return parsed["content"]

            if parsed["type"] == "tool_calls":
                tool_calls = parsed["content"]

                assistant_message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
                self.history.append(assistant_message)

                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"]["arguments"]
                    call_id = tool_call.get("id", "unknown")

                    if isinstance(raw_args, str):
                        try:
                            arguments = json.loads(raw_args)
                        except json.JSONDecodeError:
                            arguments = {"error": f"Could not parse arguments: {raw_args}"}
                    else:
                        arguments = raw_args

                    self._log(f"Tool call: {func_name}({json.dumps(arguments)[:100]})")

                    result = execute_tool(func_name, arguments)

                    self._log(f"Tool result: {result[:200]}")

                    self.trace.append({
                        "round": round_num,
                        "type": "tool_call",
                        "tool": func_name,
                        "arguments": arguments,
                        "result": result,
                    })

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result,
                    })

        self._log(f"Hit max tool rounds ({MAX_TOOL_ROUNDS}). Forcing final answer.")
        return self._force_final_answer()

    def _force_final_answer(self):
        self.history.append({
            "role": "user",
            "content": "You have reached the maximum number of tool calls. Based on what you have so far, give your best final answer now.",
        })

        api_result = llm_client.chat(messages=self.history, tools=None)
        parsed = llm_client.extract_response(api_result)

        if parsed["type"] == "text":
            return parsed["content"]
        return f"Agent could not produce a final answer. Last response: {parsed}"

    def get_trace(self):
        return self.trace

    def _log(self, message):
        if VERBOSE:
            print(f"[{self.name}] {message}")