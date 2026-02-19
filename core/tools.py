# ============================================================
# FILE: core/tools.py
# ============================================================
# Three parts:
#   1. Tool implementations — actual Python functions
#   2. Tool schemas — JSON descriptions for the LLM
#   3. Dispatcher — routes a tool call to the correct function
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
# 2. Tool schemas — tells the LLM what tools exist and what arguments they take
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression. Supports +, -, *, /, parentheses, and decimal numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '(15 + 27) * 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file and return its text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file and parent directories if they don't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Execute Python code and return the output. Use this for calculations, data processing, or testing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. (Not yet implemented — placeholder for later phase.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


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


def get_tool_schemas(tool_names: list = None) -> list:
    if tool_names is None:
        return TOOL_SCHEMAS

    return [s for s in TOOL_SCHEMAS if s["function"]["name"] in tool_names]