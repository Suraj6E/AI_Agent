# AI Agents: From Zero to Hero

A progressive, hands-on tutorial for building AI agents from scratch.

## What is an AI Agent?

An AI agent is an **LLM in a loop** that can:
1. **Observe** → receive input/context
2. **Think** → reason about what to do  
3. **Act** → execute tools/functions
4. **Repeat** → until task is complete

```
┌─────────────────────────────────────┐
│           USER TASK                 │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│         LLM (Brain)                 │◄──────┐
│   - Understands task                │       │
│   - Decides next action             │       │
│   - Interprets results              │       │
└──────────────┬──────────────────────┘       │
               ▼                              │
┌─────────────────────────────────────┐       │
│         TOOLS                       │       │
│   - web_search()                    │       │
│   - read_file()                     │       │
│   - write_code()                    │       │
│   - execute_bash()                  │───────┘
└─────────────────────────────────────┘  (results fed back)
```

## Project Structure

```
agent-tutorial/
├── README.md                      # This file
├── requirements.txt               # All dependencies
│
├── level-01-simple-agent/         # Basic tool-calling agent
│   ├── README.md
│   ├── agent_claude.py            # Claude API version
│   └── agent_local.py             # Ollama/local LLM version
│
├── level-02-react-agent/          # ReAct pattern (Reasoning + Acting)
│   ├── README.md
│   └── ...
│
├── level-03-coding-agent/         # Agent that can write & run code
│   ├── README.md
│   └── ...
│
└── level-04-multi-agent/          # Multiple agents working together
    ├── README.md
    └── ...
```

## Learning Path

| Level | Name | What You'll Learn |
|-------|------|-------------------|
| 01 | Simple Agent | Basic agent loop, tool calling, Claude API vs Local LLM |
| 02 | ReAct Agent | Reasoning traces, chain-of-thought, better decision making |
| 03 | Coding Agent | File I/O, code execution, sandboxing, real-world tasks |
| 04 | Multi-Agent | Agent orchestration, delegation, specialized roles |

## Prerequisites

- Python 3.10+
- Basic understanding of APIs
- (Optional) Ollama for local LLM testing

## Quick Start

```bash
# Clone or download this project
cd agent-tutorial

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key (for Claude version)
export ANTHROPIC_API_KEY="your-key-here"

# Start with Level 01
cd level-01-simple-agent
python agent_claude.py
```

## API Key Setup

### Claude API (Recommended for learning)
1. Go to https://console.anthropic.com
2. Create an account (separate from claude.ai)
3. You get ~$5 free credits for testing
4. Generate an API key in the dashboard
5. Set it as environment variable:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

### Local LLM (Free, runs on your machine)
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. No API key needed!

## Tips for Learning

1. **Read the code comments** - Each file is heavily documented
2. **Run the examples** - Don't just read, execute and observe
3. **Modify and experiment** - Change tools, prompts, models
4. **Check the logs** - Observe how the agent "thinks"

## Cost Estimation

For learning purposes with Claude API:
- Simple agent task: ~2,000-10,000 tokens ≈ $0.01-0.05
- Your $5 free credits = hundreds of experiments

---

Ready? Start with [Level 01: Simple Agent](./level-01-simple-agent/README.md)
