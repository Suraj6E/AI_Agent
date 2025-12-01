"""
Level 01: Simple Tool-Calling Agent (Local LLM via Ollama)
===========================================================
Same concept as Claude version, but with a local model.

Key difference: Local LLMs don't have native tool calling,
so we use prompt engineering to make the LLM output JSON.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3.2
    3. pip install ollama

Run with:
    python agent_local.py
"""

import json
import re
from datetime import datetime

try:
    import ollama
except ImportError:
    print("❌ Error: ollama package not installed!")
    print("\nTo fix this:")
    print("  pip install ollama")
    print("\nAlso make sure Ollama is installed: https://ollama.ai")
    exit(1)


# =============================================================================
# TOOLS: Same functions as Claude version
# =============================================================================

def get_current_time(timezone: str = "UTC") -> str:
    """Returns current date and time."""
    now = datetime.now()
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


def calculate(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


def search_knowledge(query: str) -> str:
    """Simulated knowledge search."""
    knowledge_base = {
        "nepal": "Nepal is a landlocked country in South Asia, located in the Himalayas. Capital: Kathmandu. Population: ~30 million.",
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "agent": "An AI agent is an LLM that can use tools and take actions in a loop to accomplish tasks.",
        "anthropic": "Anthropic is an AI safety company founded in 2021. Created the Claude family of AI models.",
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    return f"No information found for: {query}"


TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
}


# =============================================================================
# SYSTEM PROMPT: Teaches the LLM how to use tools via JSON
# =============================================================================

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

AVAILABLE TOOLS:
1. get_current_time(timezone="UTC") - Get current date and time
2. calculate(expression) - Evaluate math expressions like "2+2" or "100*0.15"
3. search_knowledge(query) - Search for information on a topic

IMPORTANT RULES:
- If you need to use a tool, respond ONLY with this JSON format:
  {"tool": "tool_name", "args": {"param": "value"}}

- If you can answer WITHOUT tools, respond with normal text.

- Use ONE tool at a time. Wait for the result before using another.

- After receiving a tool result, provide your final answer.

EXAMPLES:

User: What time is it?
Assistant: {"tool": "get_current_time", "args": {"timezone": "UTC"}}

User: What is 25 * 4?
Assistant: {"tool": "calculate", "args": {"expression": "25 * 4"}}

User: Tell me about Python
Assistant: {"tool": "search_knowledge", "args": {"query": "python"}}

User: Hello!
Assistant: Hello! How can I help you today?

User: What's 2+2?
[Tool result: 2 + 2 = 4]
Assistant: 2 + 2 equals 4.
"""


# =============================================================================
# JSON PARSER: Extract tool calls from LLM output
# =============================================================================

def parse_tool_call(response_text: str) -> dict | None:
    """
    Try to extract a tool call JSON from the LLM response.
    
    Returns:
        dict with 'tool' and 'args' if found, None otherwise
    """
    response_text = response_text.strip()
    
    if not response_text.startswith("{"):
        return None
    
    try:
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            parsed = json.loads(json_match.group())
            if "tool" in parsed:
                return parsed
    except json.JSONDecodeError:
        pass
    
    return None


# =============================================================================
# AGENT LOOP: Core logic for local LLM
# =============================================================================

def run_agent(
    user_message: str, 
    model: str = "llama3.2", 
    max_iterations: int = 10,
    verbose: bool = True
) -> str:
    """
    Agent loop for local LLM.
    
    Args:
        user_message: The task from the user
        model: Ollama model name (llama3.2, mistral, qwen2.5, etc.)
        max_iterations: Safety limit
        verbose: Whether to print debug information
    
    Returns:
        The agent's final response
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 USER TASK: {user_message}")
        print(f"🤖 MODEL: {model}")
        print(f"{'='*60}\n")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        if verbose:
            print(f"--- Iteration {iteration} ---")
        
        try:
            response = ollama.chat(model=model, messages=messages)
            llm_output = response["message"]["content"]
        except Exception as e:
            error_msg = str(e)
            if "connection refused" in error_msg.lower():
                return "Error: Cannot connect to Ollama. Make sure it's running with: ollama serve"
            elif "not found" in error_msg.lower():
                return f"Error: Model '{model}' not found. Pull it with: ollama pull {model}"
            else:
                return f"Error calling Ollama: {e}"
        
        if verbose:
            preview = llm_output[:100] + "..." if len(llm_output) > 100 else llm_output
            print(f"LLM Output: {preview}")
        
        tool_call = parse_tool_call(llm_output)
        
        if tool_call is None:
            if verbose:
                print(f"\n{'='*60}")
                print("✅ AGENT COMPLETE (no tool call)")
                print(f"{'='*60}")
            return llm_output
        
        tool_name = tool_call.get("tool")
        tool_args = tool_call.get("args", {})
        
        if verbose:
            print(f"  🔧 Tool: {tool_name}")
            print(f"     Args: {tool_args}")
        
        if tool_name in TOOL_FUNCTIONS:
            result = TOOL_FUNCTIONS[tool_name](**tool_args)
        else:
            result = f"Error: Unknown tool '{tool_name}'"
        
        if verbose:
            print(f"     Result: {result}\n")
        
        messages.append({"role": "assistant", "content": llm_output})
        messages.append({
            "role": "user", 
            "content": f"Tool result: {result}\n\nNow provide your final answer to the user based on this result."
        })
    
    return "Error: Agent reached maximum iterations."


# =============================================================================
# MAIN: Run example tasks
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 LEVEL 01: Simple Tool-Calling Agent (Local LLM)")
    print("="*60)
    
    print("\n📋 Prerequisites:")
    print("  1. Ollama installed: https://ollama.ai")
    print("  2. Model pulled: ollama pull llama3.2")
    print("  3. Ollama running: ollama serve")
    
    test_tasks = [
        "What time is it right now?",
        "Calculate 157 * 23",
        "What can you tell me about Nepal?",
        "What's 15% of 250?",
    ]
    
    available_models = ["llama3.2", "mistral", "qwen2.5", "llama3.1"]
    
    print("\nChoose a test task:")
    for i, task in enumerate(test_tasks, 1):
        print(f"  {i}. {task}")
    print(f"  5. Enter custom task")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "5":
            user_task = input("Enter your task: ").strip()
        elif choice in ["1", "2", "3", "4"]:
            user_task = test_tasks[int(choice) - 1]
        else:
            user_task = test_tasks[0]
        
        model_choice = input(f"\nModel [{available_models[0]}]: ").strip()
        model = model_choice if model_choice else available_models[0]
        
        result = run_agent(user_task, model=model)
        print(f"\n📝 FINAL ANSWER:\n{result}")
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
