# Multi-Agent AI System with GLM-4-9B
### Complete Documentation & Implementation Plan

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware & Model Selection](#2-hardware--model-selection)
3. [Understanding LLMs — Parameters & Intelligence](#3-understanding-llms--parameters--intelligence)
4. [Agent Types — How They Work](#4-agent-types--how-they-work)
5. [Multi-Agent Architecture](#5-multi-agent-architecture)
6. [GLM-4-9B Capabilities & Honest Limitations](#6-glm-4-9b-capabilities--honest-limitations)
7. [Fine-Tuning — When & Why](#7-fine-tuning--when--why)
8. [Implementation Plan](#8-implementation-plan)
9. [Project File Structure](#9-project-file-structure)
10. [Technology Stack](#10-technology-stack)

---

## 1. Project Overview

This project builds a **local, offline, private multi-agent AI system** powered by GLM-4-9B running on your personal hardware via vLLM. The system uses pure Python with no external AI frameworks — every component is transparent and under your control.

The core idea: an Orchestrator agent receives a complex task, breaks it into subtasks, delegates to specialized agents, and merges the results into a final answer. Each specialist agent internally uses a ReAct loop (Reasoning + Acting) to work through its subtask step by step.

**Why local and offline?**
- No data leaves your machine — full privacy
- No per-token API cost after hardware
- Full control over the model — can fine-tune on your own data
- Works without internet (except tools that explicitly need it, like web search)

---

## 2. Hardware & Model Selection

### Your Hardware
| Component | Spec | Notes |
|---|---|---|
| RAM | 32GB | Sufficient for model + system overhead |
| GPU VRAM | 16GB | Fits GLM-4-9B in full float16 precision |
| CPU | Latest | Good fallback for non-GPU layers |

### Why GLM-4-9B (not GLM-5)

GLM-5 was considered first but rejected for this hardware. The reason:

- GLM-5 has 744B parameters and requires 8 GPUs (tensor-parallel-size 8) for proper deployment
- The only way to run GLM-5 on a single 16GB GPU is via heavy GGUF quantization (Q2/Q3), which severely degrades quality and runs very slowly due to CPU offloading
- GLM-4-9B fits perfectly on 16GB VRAM in full float16 — fast, stable, and no quality loss from quantization

**GLM-4-9B is the right model for this hardware.**

### Why Not Claude or GPT-4 API?

Claude and GPT-4 class models are more capable for complex reasoning, but:
- Cost per token adds up quickly for an agent system that makes many calls
- Your data (prompts, tool results, code) leaves your machine
- You cannot fine-tune them on your own data
- They require internet

The architecture is designed so you can swap the model backend later if needed. If you want Claude for specific heavy reasoning tasks, you can route just the orchestrator to Claude while keeping specialists on GLM-4-9B.

---

## 3. Understanding LLMs — Parameters & Intelligence

### What Parameters Are

Parameters are the learned connections (weights) inside a neural network — analogous to synapses in a brain. They are numbers that get adjusted during training to make the model better at predicting the next token.

GLM-4-9B has 9 billion such numbers. Claude Sonnet 4.6 has an estimated 200B+.

### What More Parameters Buy You

**Pattern storage capacity** — more parameters means the model memorized more relationships between concepts. A 9B model has retained far fewer patterns than a 200B model from training.

**Reasoning depth** — larger models can hold more computational "context" internally while working through a problem. This is why big models handle multi-step reasoning more reliably.

**Generalization** — bigger models are better at applying knowledge to genuinely novel situations. Smaller models tend to be more rigid and pattern-matchy.

**Nuance and calibration** — knowing when to say "it depends," handling ambiguity, and knowing the limits of their own knowledge.

### But Parameters Are Not Everything

| Factor | What It Affects |
|---|---|
| Parameter count | The ceiling of what is possible |
| Training data quality | What the model actually learned |
| Training method (RLHF, RL) | How well it follows instructions and reasons |
| Quantization | Reduces numerical precision, slightly reduces quality |
| Context window | How much the model can "see" at once |

A well-trained 9B model beats a poorly trained 50B model on specific tasks. And a 9B model fine-tuned on your specific codebase can outperform a general 200B model on that exact domain.

### The Simple Analogy

Parameters are like RAM. More RAM lets you run more complex programs. But RAM alone doesn't make software better — the quality of the code (training data and method) matters equally. A well-written program on 16GB RAM can outperform a bloated program on 64GB RAM for the right task.

---

## 4. Agent Types — How They Work

### Type 1: Tool-Calling Agent

The simplest agent pattern. One LLM in a loop with access to tools (functions). The LLM decides when to call a tool, what arguments to pass, and interprets the result.

```
User Input
    ↓
LLM sees available tools
    ↓
Decides: "I need tool X"
    ↓
Tool executes → returns result
    ↓
LLM formulates final answer
```

Best for: single focused tasks where the right tool is obvious. Example: "What is 157 * 23?" → calls calculate tool.

---

### Type 2: ReAct Agent (Reasoning + Acting)

Adds an explicit **Thought step** before every action. The model reasons out loud about what it knows, what it needs, and why it's choosing a particular tool before acting.

```
User Input
    ↓
[Thought: what do I know? what do I need?]
[Act: call tool]
[Observe: read result]
    ↓
[Thought: do I have enough? what next?]
[Act: call another tool or finish]
[Observe: read result]
    ↓
... repeat until done ...
    ↓
Final Answer
```

Why it's better than plain tool-calling: forcing explicit reasoning before each action catches mistakes early, prevents the model from jumping to conclusions, and makes the agent's behavior debuggable — you can read the Thought steps and see exactly where it went wrong.

Best for: complex multi-step tasks requiring sequential reasoning. Example: "Research the latest papers on neuro-symbolic AI and write a summary."

---

### Type 3: Multi-Agent System

Multiple specialized LLM instances, each with their own role, system prompt, tools, and memory — coordinated by a central Orchestrator.

```
User Input
    ↓
Orchestrator
(plans the work, decides which agents are needed)
    ↓
    ├── Researcher Agent    (ReAct loop internally)
    ├── Coder Agent         (ReAct loop internally)
    └── Reviewer Agent      (ReAct loop internally)
         ↓
Orchestrator receives all results
    ↓
Merges into Final Answer
```

Each specialist agent is itself a ReAct agent — so multi-agent is ReAct agents coordinated by an orchestrator.

Why multi-agent genuinely helps (even with the same base model):

- **Focused context per agent** — each agent's prompt is smaller and domain-specific. LLMs perform measurably better with focused, constrained prompts than open-ended complex ones
- **Separation of concerns** — a researcher isn't distracted by coding concerns
- **Error catching via reviewer** — a fresh agent reviewing an output catches errors the writer missed
- **Longer effective reasoning chain** — the chain of agents can reason across more steps than any single agent's context window allows
- **Parallel execution** — researcher and coder can work simultaneously on different subtasks

The honest limitation: multi-agent raises how consistently you operate near the model's intelligence ceiling, but does not raise the ceiling itself. For complex deep reasoning tasks, all agents hit the same wall.

---

## 5. Multi-Agent Architecture

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER TASK                            │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR                           │
│  - Receives task                                        │
│  - Produces JSON plan with subtasks                     │
│  - Delegates to specialists                             │
│  - Merges results into final answer                     │
└──────┬──────────────────┬──────────────────┬────────────┘
       ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│  RESEARCHER  │  │    CODER     │  │    REVIEWER      │
│              │  │              │  │                  │
│ Tools:       │  │ Tools:       │  │ Tools:           │
│ - web_search │  │ - run_code   │  │ - read_file      │
│ - read_file  │  │ - write_file │  │ - run_code       │
│              │  │ - read_file  │  │                  │
│ ReAct loop   │  │ ReAct loop   │  │ ReAct loop       │
└──────────────┘  └──────────────┘  └──────────────────┘
       ↓                  ↓                  ↓
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATOR MERGES RESULTS                │
└─────────────────────────────────────────────────────────┘
                         ↓
                   FINAL ANSWER
```

### Agent Roles

**Orchestrator**
- Receives the raw user task
- Reasons about what subtasks are needed
- Produces a structured plan (JSON) specifying which agent handles what
- After all agents complete, instructs the LLM to merge results
- Does NOT do the actual research or coding itself

**Researcher Agent**
- Specializes in finding, reading, and summarizing information
- Primary tools: web_search, read_file
- Uses ReAct internally: searches → reads → decides if enough → searches more if needed
- Returns a structured summary to the orchestrator

**Coder Agent**
- Specializes in writing, running, and debugging Python code
- Primary tools: run_python_code, write_file, read_file
- Uses ReAct internally: writes code → runs it → reads error → fixes → runs again
- Returns working code + output to the orchestrator

**Reviewer Agent** *(planned for Phase 2)*
- Receives the output from researcher or coder
- Checks for errors, logical gaps, or code bugs
- Returns a review verdict: pass, or specific feedback to fix
- Creates a feedback loop — coder fixes based on reviewer's notes

---

## 6. GLM-4-9B Capabilities & Honest Limitations

### What It Does Well

- Code generation for clear, scoped tasks (write a function, fix a bug, explain code)
- Following structured output instructions (JSON, markdown)
- Bilingual — English and Chinese natively
- Tool calling — understands when to call tools and formats arguments correctly
- Summarization and Q&A over provided documents
- Multi-turn conversation with memory

### Where It Struggles Compared to Frontier Models (Claude, GPT-4)

- Complex multi-step reasoning — more likely to lose track mid-chain
- System design and architecture decisions — tends toward generic answers
- Math-heavy code (ML algorithms, numerical methods) — inconsistent accuracy
- Deep domain expertise — no specialty, jack of all trades
- Self-awareness of its own errors — less likely to catch its own mistakes

### Honest Comparison: GLM-4-9B vs Claude Sonnet 4.6 for Coding

Claude Sonnet 4.6 wins significantly. The parameter gap alone (9B vs estimated 200B+) means Claude handles complex, multi-file, architecturally nuanced code far better.

The only reasons to choose GLM-4-9B over Claude for coding:
- Your code is private and cannot leave your machine
- Cost — Claude API charges per token, GLM-4-9B is free after hardware
- You want to fine-tune it on your specific codebase

---

## 7. Fine-Tuning — When & Why

### What Fine-Tuning Does

Fine-tuning takes the base GLM-4-9B weights (trained on 28.5T tokens of general internet data) and further trains them on your specific domain data. It shifts the model's probability distribution toward your domain without erasing general knowledge.

It does not make the model smarter in general. It makes it more accurate and consistent within your specific domain.

### The "Bias" Question

Fine-tuning does introduce bias — intentionally. This is the goal. You are trading broad mediocrity for narrow excellence.

This only becomes a problem when:

| Problem | Cause | Solution |
|---|---|---|
| Overfitting | Training data too small | Use 1000+ examples minimum |
| Catastrophic forgetting | Data too narrow | Mix 80% domain + 20% general data |
| Learning bad patterns | Low quality data | Curate carefully — 500 good examples beats 5000 bad ones |
| Brittle on edge cases | Not enough variety | Include diverse examples within your domain |

### When Fine-Tuning Makes Sense for This Project

Fine-tuning GLM-4-9B on your own data is worth doing once you have:
- A clear domain (trading code, neuro-symbolic AI research, specific coding patterns)
- At least 500-1000 high quality examples in that domain
- A way to evaluate improvement (a test set that measures what you care about)

Tools for fine-tuning: Hugging Face `trl` library or Unsloth (uses less VRAM, faster).

---

## 8. Implementation Plan

### Phase 1: Foundation (Get Everything Running)

**Goal:** vLLM serving GLM-4-9B locally, basic agent loop working end-to-end.

Step 1 — Environment setup
- Install Python dependencies: `vllm`, `transformers`, `huggingface_hub`, `torch`, `accelerate`
- Set up project folder structure
- Create `.env` file for any configuration

Step 2 — Download GLM-4-9B
- Use `huggingface_hub` to download `THUDM/glm-4-9b-chat` to `./models/glm-4-9b-chat`
- Prefer safetensors format, exclude `.pt` and `.bin` files
- Verify download integrity

Step 3 — vLLM server
- Serve from local model folder with `--trust-remote-code` flag
- Set `--served-model-name glm-4-9b` for clean API naming
- Set `--max-model-len 8192` to stay within VRAM budget
- Verify with a simple curl test

Step 4 — Basic LLM client
- `core/llm_client.py` — handles all HTTP communication with vLLM
- Implement `chat()` function with messages, tools, and temperature parameters
- Add basic error handling for connection failures

Step 5 — Tool definitions
- `core/tools.py` — implement and test each tool function independently
- Implement tool schemas (JSON) for each tool
- Implement `execute_tool()` dispatcher
- Test each tool function in isolation before connecting to the agent

Step 6 — Base agent class
- `core/agent.py` — the loop that handles tool calls
- Implement history management for multi-turn memory
- Add max_tool_rounds safety limit
- Test with a simple single-agent task

**Completion check:** run a single agent that calls two tools in sequence and returns a correct answer.

---

### Phase 2: ReAct Upgrade

**Goal:** All agents reason explicitly before every action.

Step 1 — Update system prompts
- Add Thought/Act/Observe format instructions to every agent's system prompt
- The model should output its reasoning before deciding to call a tool
- Reasoning trace should be visible in verbose mode

Step 2 — Update agent loop
- Parse and display Thought steps separately from tool calls in verbose output
- Store thought traces in history for debugging
- Add a thought log to each agent for post-run analysis

Step 3 — Test and compare
- Run the same task with and without ReAct
- Compare output quality and error rate
- Document which task types benefit most from ReAct

**Completion check:** agent produces visible Thought → Act → Observe cycles. Errors are catchable by reading the thought trace.

---

### Phase 3: Multi-Agent Orchestration

**Goal:** Orchestrator delegates tasks to Researcher and Coder, merges results.

Step 1 — Researcher agent
- `agents/researcher.py` — system prompt focused on information gathering
- Tools: web_search, read_file
- Inherits from base Agent with ReAct loop
- Returns structured summary

Step 2 — Coder agent
- `agents/coder.py` — system prompt focused on code quality and testing
- Tools: run_python_code, write_file, read_file
- Self-corrects on errors — runs code, reads error, fixes, runs again
- Returns working code + execution output

Step 3 — Orchestrator
- `agents/orchestrator.py` — does not do research or coding itself
- Prompts LLM to produce a JSON plan with subtasks and agent assignments
- Handles JSON parsing with fallback for malformed output
- Runs subtasks (sequentially first, parallel later)
- Merges results via a final LLM call

Step 4 — Entry point
- `run.py` — interactive loop, feeds user input to orchestrator
- Display which agent is working on what in real time
- Show final merged answer cleanly

**Completion check:** a task requiring both research and coding is correctly split, delegated, and merged into a coherent answer.

---

### Phase 4: Reviewer Agent + Feedback Loop

**Goal:** Output quality improves through self-correction cycles.

Step 1 — Reviewer agent
- `agents/reviewer.py` — system prompt focused on finding errors and gaps
- Reviews code output: does it run? does it do what was asked?
- Reviews research output: is it complete? are there obvious gaps?
- Returns a structured verdict: PASS or FEEDBACK with specific issues

Step 2 — Feedback loop in orchestrator
- If reviewer returns FEEDBACK, send it back to the relevant agent
- Agent corrects based on feedback and resubmits
- Limit to 2-3 correction rounds to avoid infinite loops
- Track correction rounds in logs

Step 3 — Evaluation
- Build a small test set of tasks with known correct answers
- Measure pass rate with and without reviewer loop
- Decide if the reviewer adds enough value to justify the extra LLM calls

**Completion check:** a task with a deliberate bug in the code gets caught by the reviewer and corrected by the coder without user intervention.

---

### Phase 5: Fine-Tuning (Optional, Later)

**Goal:** Improve GLM-4-9B accuracy on your specific domain.

Step 1 — Data collection
- Identify your target domain (trading code, research summaries, etc.)
- Collect or generate 1000+ high quality input/output examples
- Mix in ~20% general examples to prevent catastrophic forgetting
- Split into train/validation/test sets

Step 2 — Fine-tuning setup
- Use Unsloth for memory-efficient training on your 16GB GPU
- LoRA fine-tuning — trains only a small adapter on top of frozen weights, much less VRAM than full fine-tune
- Monitor validation loss to catch overfitting early

Step 3 — Evaluation
- Test on your held-out test set
- Compare against base GLM-4-9B on the same tasks
- Only deploy the fine-tuned model if it measurably improves on your target domain without degrading general tasks

---

## 9. Project File Structure

```
multi_agent/
│
├── README.md                    ← This document
├── .env                         ← Configuration (model path, vLLM URL, etc.)
├── requirements.txt             ← All Python dependencies
├── run.py                       ← Entry point, interactive loop
│
├── models/
│   └── glm-4-9b-chat/           ← Downloaded model weights (local)
│
├── core/
│   ├── llm_client.py            ← vLLM HTTP communication
│   ├── agent.py                 ← Base agent class with ReAct loop
│   └── tools.py                 ← Tool functions, schemas, executor
│
├── agents/
│   ├── orchestrator.py          ← Plans, delegates, merges
│   ├── researcher.py            ← Information gathering specialist
│   ├── coder.py                 ← Code writing and testing specialist
│   └── reviewer.py              ← Output validation (Phase 4)
│
├── logs/
│   └── agent_traces/            ← Thought/Act/Observe traces per run
│
└── tests/
    ├── test_tools.py            ← Unit tests for each tool function
    ├── test_agents.py           ← Integration tests per agent
    └── eval_tasks.json          ← Test tasks with expected outputs
```

---

## 10. Technology Stack

| Component | Technology | Why |
|---|---|---|
| LLM | GLM-4-9B | Fits 16GB VRAM, MIT license, good tool calling |
| Inference server | vLLM | OpenAI-compatible API, fast, handles batching |
| Agent framework | Pure Python | Full transparency, no framework lock-in |
| Model download | huggingface_hub | Official HuggingFace download tool |
| HTTP client | requests | Simple, no overhead |
| Code execution tool | subprocess | Built-in Python, no extra dependencies |
| Fine-tuning (Phase 5) | Unsloth + LoRA | Memory efficient, works on 16GB GPU |
| Vector DB for RAG (future) | ChromaDB or Qdrant | Local, no cloud dependency |

---

## Quick Reference — Key Decisions Made

| Decision | Choice | Reason |
|---|---|---|
| GLM-5 vs GLM-4-9B | GLM-4-9B | GLM-5 requires 8 GPUs, too large for hardware |
| Framework vs Pure Python | Pure Python | Full control, no abstraction hiding bugs |
| Tool-calling vs ReAct vs Multi-agent | Multi-agent with ReAct internally | Best output quality for complex tasks |
| Fine-tune vs RAG for domain knowledge | RAG first, fine-tune later | RAG is faster to implement, fine-tune when data is ready |
| Local vs API model | Local | Privacy, cost, and fine-tuning control |