# ============================================================
# FILE: tests/test_tools.py
# ============================================================
# Run this to verify all tools work before connecting to the LLM.
#
# Usage: python -m tests.test_tools  (from project root)
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tools import execute_tool


def test_calculate():
    assert execute_tool("calculate", {"expression": "2 + 3"}) == "5"
    assert execute_tool("calculate", {"expression": "(10 + 5) * 2"}) == "30"
    assert "Error" in execute_tool("calculate", {"expression": "import os"})
    print("  calculate: PASSED")


def test_read_file():
    result = execute_tool("read_file", {"file_path": "requirements.txt"})
    assert "requests" in result
    assert "Error" in execute_tool("read_file", {"file_path": "nonexistent_file.txt"})
    print("  read_file: PASSED")


def test_write_file():
    test_path = os.path.join(os.environ.get("TEMP", "/tmp"), "test_write_tool.txt")
    result = execute_tool("write_file", {"file_path": test_path, "content": "hello test"})
    assert "Successfully" in result
    read_back = execute_tool("read_file", {"file_path": test_path})
    assert read_back == "hello test"
    os.remove(test_path)
    print("  write_file: PASSED")


def test_run_python_code():
    result = execute_tool("run_python_code", {"code": "print(2 + 2)"})
    assert "4" in result
    result_err = execute_tool("run_python_code", {"code": "raise ValueError('test')"})
    assert "ValueError" in result_err
    print("  run_python_code: PASSED")


def test_web_search():
    result = execute_tool("web_search", {"query": "test query"})
    assert "not implemented" in result.lower()
    print("  web_search: PASSED (stub)")


def test_unknown_tool():
    result = execute_tool("nonexistent_tool", {})
    assert "Error" in result
    print("  unknown tool handling: PASSED")


if __name__ == "__main__":
    print("Running tool tests...\n")
    test_calculate()
    test_read_file()
    test_write_file()
    test_run_python_code()
    test_web_search()
    test_unknown_tool()
    print("\nAll tool tests passed.")