# ============================================================
# FILE: agents/orchestrator.py
# ============================================================
# Phase 3: Orchestrator
#
# Receives a user task and:
#   1. Asks the LLM to produce a JSON plan with subtasks
#   2. Parses the plan
#   3. Delegates each subtask to the right specialist agent
#   4. Collects all results
#   5. Asks the LLM to merge results into a final answer
#
# The orchestrator does NOT do research or coding itself.
# It only plans, delegates, and merges.
#
# Specialist agents available:
#   - researcher: information gathering (web_search, read_file)
#   - coder: code writing and testing (run_python_code, write_file, read_file)
#   - general: any task, all tools (fallback)
#
# If the plan has only one subtask, it still delegates — the
# orchestrator never does the work directly.
# ============================================================
#
# Changes:
#   - Errors during planning and delegation are now recorded in self.trace
#     so trace files are never empty on failure.
#   - Subtask errors are recorded with type "error" for easy filtering.
# ============================================================

import json
import re
import os
from datetime import datetime
from dotenv import load_dotenv
from core import llm_client
from core.agent import Agent
from agents.researcher import create_researcher
from agents.coder import create_coder

load_dotenv()

VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Planning prompt — tells the LLM how to produce a plan
# ---------------------------------------------------------------------------

PLAN_PROMPT = """You are an Orchestrator. Your job is to break a user's task into subtasks and assign each to the right specialist agent.

Available agents:
- researcher: Finds and summarizes information. Has tools: web_search, read_file.
- coder: Writes, runs, and debugs Python code. Has tools: run_python_code, write_file, read_file.
- general: Handles any task that does not clearly fit researcher or coder.

RULES:
1. Analyze the user's task and decide what subtasks are needed.
2. Output a JSON plan — ONLY the JSON, nothing else.
3. Each subtask must have: "id" (number), "agent" (string), "task" (string).
4. Subtasks run in order. Later subtasks can say "using the result from subtask 1".
5. Use the fewest subtasks necessary. Simple tasks may need only 1.
6. If the task is a simple question you can answer directly, use one subtask with agent "general".

OUTPUT FORMAT (JSON only, no markdown, no explanation):
{
  "subtasks": [
    {"id": 1, "agent": "researcher", "task": "Find information about X"},
    {"id": 2, "agent": "coder", "task": "Using the research results, write code to do Y"}
  ]
}"""


# ---------------------------------------------------------------------------
# Merge prompt — tells the LLM how to combine results
# ---------------------------------------------------------------------------

MERGE_PROMPT_TEMPLATE = """You are an Orchestrator merging results from specialist agents.

The user's original task was:
{user_task}

Here are the results from each subtask:

{results_text}

Your job: Combine these results into one clear, complete final answer for the user.
- Do NOT repeat the subtask structure, just give the final answer.
- If a subtask failed or returned an error, mention it briefly.
- Be direct and concise."""


# ---------------------------------------------------------------------------
# General-purpose agent (fallback for tasks that aren't clearly research or code)
# ---------------------------------------------------------------------------

GENERAL_PROMPT = """You are a helpful general-purpose assistant.
Answer the user's question clearly and directly.
Use tools if needed, or answer from your knowledge if you can."""


def create_general(name="General"):
    return Agent(
        name=name,
        system_prompt=GENERAL_PROMPT,
        tool_names=["calculate", "read_file", "write_file", "run_python_code", "web_search"],
    )


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------

def parse_plan(raw_text):
    """
    Extract the JSON plan from the LLM's output.
    Tries to find a JSON object with a "subtasks" key.
    Returns a list of subtask dicts, or None if parsing fails.
    """
    # Strip <think>...</think> if present (DeepSeek-R1)
    clean = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    # Try to find JSON in the text
    json_match = re.search(r'\{.*\}', clean, re.DOTALL)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group())
        subtasks = parsed.get("subtasks")
        if isinstance(subtasks, list) and len(subtasks) > 0:
            return subtasks
        return None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self):
        self.trace = []

    def run(self, user_task):
        self._log("=" * 50)
        self._log("ORCHESTRATOR STARTED")
        self._log(f"Task: {user_task}")
        self._log("=" * 50)

        self.trace = []

        # --- Step 1: Plan ---
        self._log("\n[Step 1] Planning...")

        plan_messages = [
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": user_task},
        ]
        plan_raw = llm_client.chat(messages=plan_messages, temperature=0.3)

        if plan_raw.startswith("[ERROR]"):
            self._log(f"Planning failed: {plan_raw}")
            self.trace.append({
                "step": "plan",
                "type": "error",
                "error": plan_raw,
            })
            return f"Orchestrator error during planning: {plan_raw}"

        self._log(f"Raw plan:\n{plan_raw}\n")

        subtasks = parse_plan(plan_raw)

        if subtasks is None:
            self._log("Failed to parse plan. Falling back to single general agent.")
            subtasks = [{"id": 1, "agent": "general", "task": user_task}]

        self._log(f"Plan: {len(subtasks)} subtask(s)")
        for st in subtasks:
            self._log(f"  [{st['id']}] {st['agent']}: {st['task'][:80]}")

        self.trace.append({
            "step": "plan",
            "raw": plan_raw,
            "subtasks": subtasks,
        })

        # --- Step 2: Delegate ---
        self._log("\n[Step 2] Delegating to specialists...")

        results = []
        for st in subtasks:
            subtask_id = st.get("id", "?")
            agent_type = st.get("agent", "general")
            task_text = st.get("task", user_task)

            self._log(f"\n{'─'*40}")
            self._log(f"Subtask {subtask_id} → {agent_type}")
            self._log(f"Task: {task_text}")
            self._log(f"{'─'*40}")

            # Add context from previous results if this isn't the first subtask
            if results:
                context = self._build_context(results)
                task_with_context = (
                    f"{task_text}\n\n"
                    f"Context from previous subtasks:\n{context}"
                )
            else:
                task_with_context = task_text

            agent = self._create_agent(agent_type, subtask_id)
            result = agent.run(task_with_context)

            self._log(f"\nSubtask {subtask_id} result: {result[:200]}")

            result_entry = {
                "id": subtask_id,
                "agent": agent_type,
                "task": task_text,
                "result": result,
                "trace": agent.get_trace(),
            }

            # Tag errors so they're easy to find in trace files
            if result.startswith("Agent error:") or result.startswith("[ERROR]"):
                result_entry["type"] = "error"

            results.append(result_entry)

            self.trace.append({
                "step": f"subtask_{subtask_id}",
                "agent": agent_type,
                "task": task_text,
                "result": result,
                "agent_trace": agent.get_trace(),
            })

        # --- Step 3: Merge ---
        self._log(f"\n{'='*50}")
        self._log("[Step 3] Merging results...")

        if len(results) == 1:
            # Single subtask — no merge needed, just return the result directly
            final_answer = results[0]["result"]
            self._log("Single subtask — returning result directly.")
        else:
            final_answer = self._merge_results(user_task, results)

        self._log(f"\nFinal answer: {final_answer[:200]}")

        self.trace.append({
            "step": "final_answer",
            "answer": final_answer,
        })

        return final_answer

    def _create_agent(self, agent_type, subtask_id):
        """Create the right specialist agent based on type."""
        name = f"{agent_type.capitalize()}_{subtask_id}"

        if agent_type == "researcher":
            return create_researcher(name=name)
        elif agent_type == "coder":
            return create_coder(name=name)
        else:
            return create_general(name=name)

    def _build_context(self, results):
        """Build a context string from previous subtask results."""
        lines = []
        for r in results:
            lines.append(f"[Subtask {r['id']} — {r['agent']}]")
            lines.append(r["result"][:500])
            lines.append("")
        return "\n".join(lines)

    def _merge_results(self, user_task, results):
        """Ask the LLM to merge all subtask results into one final answer."""
        results_text = ""
        for r in results:
            results_text += f"--- Subtask {r['id']} ({r['agent']}) ---\n"
            results_text += f"Task: {r['task']}\n"
            results_text += f"Result:\n{r['result']}\n\n"

        merge_prompt = MERGE_PROMPT_TEMPLATE.format(
            user_task=user_task,
            results_text=results_text,
        )

        merge_messages = [
            {"role": "system", "content": "You merge specialist results into a clear final answer."},
            {"role": "user", "content": merge_prompt},
        ]

        merged = llm_client.chat(messages=merge_messages, temperature=0.3)

        if merged.startswith("[ERROR]"):
            # If merge fails, concatenate results as fallback
            self._log(f"Merge LLM call failed: {merged}")
            return "\n\n".join(r["result"] for r in results)

        # Strip <think> tags if present
        merged = re.sub(r'<think>.*?</think>', '', merged, flags=re.DOTALL).strip()

        return merged

    def save_trace(self, directory="logs/agent_traces"):
        """Save the full orchestrator trace to a JSON file."""
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(directory, f"Orchestrator_{timestamp}.json")

        trace_data = {
            "agent": "Orchestrator",
            "timestamp": timestamp,
            "steps": len(self.trace),
            "trace": self.trace,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, default=str)

        return filepath

    def get_trace(self):
        return self.trace

    def _log(self, message):
        if VERBOSE:
            print(f"[Orchestrator] {message}")