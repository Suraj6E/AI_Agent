# ============================================================
# FILE: core/tools.py
# ============================================================
# Three parts:
#   1. Tool implementations — actual Python functions
#   2. Tool descriptions — plain text for the system prompt
#   3. Dispatcher — routes a tool call to the correct function
#
# We no longer use JSON schemas (the old TOOL_SCHEMAS).
# Instead, we describe tools in plain text inside the system prompt
# and the model outputs a [TOOL_CALL] tag that our code parses.
# ============================================================

import subprocess
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Tool implementations
# ---------------------------------------------------------------------------

def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return f"Error: expression contains disallowed characters. Only numbers and +-*/.() are allowed."

    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def read_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        return f"Error: file not found at {file_path}"
    if not path.is_file():
        return f"Error: {file_path} is not a file"
    if path.stat().st_size > 100_000:
        return f"Error: file too large ({path.stat().st_size} bytes). Max 100KB."

    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(file_path: str, content: str) -> str:
    path = Path(file_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


def run_python_code(code: str) -> str:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = ""
        if result.stdout:
            output += f"stdout:\n{result.stdout}"
        if result.stderr:
            output += f"stderr:\n{result.stderr}"
        if not output:
            output = "(no output)"

        return output

    except subprocess.TimeoutExpired:
        return "Error: code execution timed out after 30 seconds."
    except Exception as e:
        return f"Error running code: {e}"


def web_search(query: str) -> str:
    return (
        f"[web_search is not implemented yet] "
        f"Query was: '{query}'. "
        f"This tool will be connected to a real search API in a later phase."
    )


# ---------------------------------------------------------------------------
# 2. Tool descriptions — plain text that goes into the system prompt
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS = {
    "calculate": {
        "description": "Evaluate a math expression. Supports +, -, *, /, parentheses, and decimals.",
        "arguments": {"expression": "string — the math expression, e.g. '(15 + 27) * 3'"},
    },
    "read_file": {
        "description": "Read the contents of a local file and return its text.",
        "arguments": {"file_path": "string — path to the file to read"},
    },
    "write_file": {
        "description": "Write content to a file. Creates the file and parent directories if needed.",
        "arguments": {"file_path": "string — path to write to", "content": "string — text to write"},
    },
    "run_python_code": {
        "description": "Execute Python code and return stdout/stderr. Use for calculations or testing code.",
        "arguments": {"code": "string — Python code to execute"},
    },
    "web_search": {
        "description": "Search the web for information. (Not yet implemented — placeholder.)",
        "arguments": {"query": "string — the search query"},
    },
}


def get_tool_descriptions(tool_names: list = None) -> str:
    """
    Generate a plain-text description of tools for the system prompt.
    If tool_names is None, includes all tools.
    """
    if tool_names is None:
        tool_names = list(TOOL_DESCRIPTIONS.keys())

    lines = []
    for name in tool_names:
        if name not in TOOL_DESCRIPTIONS:
            continue
        info = TOOL_DESCRIPTIONS[name]
        args_str = json.dumps(info["arguments"])
        lines.append(f"- {name}: {info['description']}")
        lines.append(f"  Arguments: {args_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Dispatcher — routes tool calls to the correct function
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "calculate": calculate,
    "read_file": read_file,
    "write_file": write_file,
    "run_python_code": run_python_code,
    "web_search": web_search,
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{tool_name}'. Available tools: {list(TOOL_FUNCTIONS.keys())}"

    try:
        func = TOOL_FUNCTIONS[tool_name]
        return func(**arguments)
    except TypeError as e:
        return f"Error: wrong arguments for tool '{tool_name}': {e}"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {e}"