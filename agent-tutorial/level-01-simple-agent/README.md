# Level 01: Simple Tool-Calling Agent

## Goal

Build the most basic AI agent pattern: **LLM + Tools in a loop**.

By the end of this level, you will understand:
- How an agent loop works
- How to define tools for an LLM
- The difference between Claude API and Local LLM tool calling
- How to execute tools and feed results back to the LLM

## The Core Concept

```
USER: "What's 157 * 23?"
         │
         ▼
┌─────────────────────────────┐
│  LLM receives message       │
│  Sees available tools       │
│  Decides: "I need calculate"│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Tool: calculate            │
│  Input: "157 * 23"          │
│  Output: "3611"             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  LLM sees result            │
│  Formulates answer          │
│  "157 * 23 = 3611"          │
└─────────────────────────────┘
```

## Files in This Level

| File | Description |
|------|-------------|
| `agent_claude.py` | Agent using Claude API (native tool support) |
| `agent_local.py` | Agent using Ollama (prompt-based tool calling) |

## Setup Instructions

### Option A: Claude API (Recommended)

1. **Get your API key:**
   - Go to https://console.anthropic.com
   - Sign up (you get ~$5 free credits)
   - Create an API key

2. **Set the environment variable:**
   ```bash
   # Linux/Mac
   export ANTHROPIC_API_KEY="sk-ant-your-key-here"
   
   # Windows (PowerShell)
   $env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
   
   # Or create a .env file (see .env.example)
   ```

3. **Install dependencies:**
   ```bash
   pip install anthropic
   ```

4. **Run:**
   ```bash
   python agent_claude.py
   ```

### Option B: Local LLM (Free)

1. **Install Ollama:**
   - Download from https://ollama.ai
   - Follow installation for your OS

2. **Pull a model:**
   ```bash
   ollama pull llama3.2
   # Or: ollama pull mistral
   # Or: ollama pull qwen2.5
   ```

3. **Install Python client:**
   ```bash
   pip install ollama
   ```

4. **Run:**
   ```bash
   python agent_local.py
   ```

## Understanding the Code

### The Agent Loop (Simplified)

```python
while True:
    # 1. Send messages to LLM (with tools available)
    response = llm.chat(messages, tools=TOOLS)
    
    # 2. Check if LLM wants to use a tool
    if response.wants_tool:
        # 3. Execute the tool
        result = execute_tool(response.tool_name, response.tool_args)
        
        # 4. Add result to messages and loop again
        messages.append(tool_result)
    else:
        # 5. LLM is done, return final answer
        return response.text
```

### Tool Definition

Each tool needs:
1. **A function** that does the actual work
2. **A schema** that tells the LLM what the tool does

```python
# The actual function
def calculate(expression: str) -> str:
    result = eval(expression)
    return f"Result: {result}"

# The schema (tells LLM how to use it)
tool_schema = {
    "name": "calculate",
    "description": "Perform math calculations",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression like '2+2'"
            }
        },
        "required": ["expression"]
    }
}
```

## Key Differences: Claude API vs Local LLM

| Aspect | Claude API | Local LLM |
|--------|-----------|-----------|
| Tool Calling | Native `tools` parameter | Prompt engineering |
| Detection | `stop_reason == "tool_use"` | Parse JSON from text |
| Reliability | High | Depends on model |
| Cost | ~$0.003 per 1K tokens | Free |
| Setup | API key only | Ollama + model download |

## Exercises

After running the examples, try these:

### Exercise 1: Add a New Tool
Add a `get_weather` tool that returns fake weather data.

```python
def get_weather(city: str) -> str:
    # Fake implementation
    return f"Weather in {city}: 22°C, Sunny"
```

Don't forget to:
- Add the function to `TOOL_FUNCTIONS`
- Add the schema to `TOOLS_SCHEMA`

### Exercise 2: Try Different Models
For local version, try different models:
```python
run_agent("What time is it?", model="mistral")
run_agent("What time is it?", model="qwen2.5")
```

### Exercise 3: Multi-Tool Task
Give the agent a task that requires multiple tools:
```
"What's 15% of 250, and what time is it now?"
```

Watch how it calls multiple tools in sequence.

## Common Issues

### "API key not found"
Make sure you've set the environment variable correctly:
```bash
echo $ANTHROPIC_API_KEY  # Should print your key
```

### "Ollama connection refused"
Make sure Ollama is running:
```bash
ollama serve  # Start the server
```

### "Model not found"
Pull the model first:
```bash
ollama pull llama3.2
```

## What's Next?

Once you're comfortable with this level, move on to:

**[Level 02: ReAct Agent](../level-02-react-agent/)** - Add explicit reasoning traces so you can see the agent "thinking" before it acts.

---

## Quick Test Commands

```bash
# Test Claude version
python agent_claude.py

# Test local version  
python agent_local.py

# Interactive mode (if implemented)
python agent_claude.py --interactive
```
