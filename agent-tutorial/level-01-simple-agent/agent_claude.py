"""
Level 01: Simple Tool-Calling Agent (Claude API)
=================================================
This is the most basic agent pattern: LLM + Tools in a loop.

Run with:
    export ANTHROPIC_API_KEY="your-key"
    python agent_claude.py
"""

import anthropic
import json
import os
from datetime import datetime


# =============================================================================
# TOOLS: Define functions the agent can call
# =============================================================================

def get_current_time(timezone: str = "UTC") -> str:
    """Returns current date and time."""
    now = datetime.now()
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    Note: eval() is used here for simplicity. In production, use a safer parser.
    """
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


def search_knowledge(query: str) -> str:
    """
    Simulated knowledge search.
    In a real agent, this would search the web, a database, or documents.
    """
    knowledge_base = {
        "nepal": "Nepal is a landlocked country in South Asia, located in the Himalayas. Capital: Kathmandu. Population: ~30 million. Known for Mount Everest.",
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991. Known for readability and versatility.",
        "agent": "An AI agent is an LLM that can use tools and take actions in a loop to accomplish tasks autonomously.",
        "anthropic": "Anthropic is an AI safety company founded in 2021. Created the Claude family of AI models.",
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    return f"No information found for: {query}"


# =============================================================================
# TOOL REGISTRY: Maps tool names to functions
# =============================================================================

TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
}


# =============================================================================
# TOOL SCHEMAS: Tells Claude what tools are available and how to use them
# =============================================================================

TOOLS_SCHEMA = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time. Use when user asks about current time or date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (default: UTC)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations. Use for any math operations like addition, multiplication, percentages, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g., '2 + 2', '15 * 7', '100 * 0.15'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "Search for information on a topic. Use when you need facts you don't know.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or question to search for",
                }
            },
            "required": ["query"],
        },
    },
]


# =============================================================================
# AGENT LOOP: The core logic that runs the agent
# =============================================================================

def run_agent(user_message: str, max_iterations: int = 10, verbose: bool = True) -> str:
    """
    Main agent loop. Takes a user message and runs until completion.
    
    Args:
        user_message: The task/question from the user
        max_iterations: Safety limit to prevent infinite loops
        verbose: Whether to print debug information
    
    Returns:
        The agent's final response
    """
    
    client = anthropic.Anthropic()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 USER TASK: {user_message}")
        print(f"{'='*60}\n")
    
    messages = [{"role": "user", "content": user_message}]
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        if verbose:
            print(f"--- Iteration {iteration} ---")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=TOOLS_SCHEMA,
            messages=messages,
        )
        
        if verbose:
            print(f"Stop reason: {response.stop_reason}")
        
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            
            if verbose:
                print(f"\n{'='*60}")
                print("✅ AGENT COMPLETE")
                print(f"{'='*60}")
            
            return final_text
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
                    
                    if verbose:
                        print(f"  🔧 Tool: {tool_name}")
                        print(f"     Input: {json.dumps(tool_input)}")
                    
                    if tool_name in TOOL_FUNCTIONS:
                        result = TOOL_FUNCTIONS[tool_name](**tool_input)
                    else:
                        result = f"Error: Unknown tool '{tool_name}'"
                    
                    if verbose:
                        print(f"     Result: {result}\n")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result,
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    return "Error: Agent reached maximum iterations without completing."


# =============================================================================
# MAIN: Run example tasks
# =============================================================================

if __name__ == "__main__":
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set!")
        print("\nTo fix this:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("\nGet your key at: https://console.anthropic.com")
        exit(1)
    
    print("\n" + "="*60)
    print("🤖 LEVEL 01: Simple Tool-Calling Agent (Claude API)")
    print("="*60)
    
    test_tasks = [
        "What time is it right now?",
        "Calculate 157 * 23 + 89",
        "What can you tell me about Nepal?",
        "What's 15% of 250, and also what time is it?",
    ]
    
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
        
        result = run_agent(user_task)
        print(f"\n📝 FINAL ANSWER:\n{result}")
        
    except KeyboardInterrupt:
        print("\n\nExiting...")
