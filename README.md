# Multi-Agent AI System — Local LLMs via Ollama
### Local, Offline, Learn-by-Building

---

## Table of Contents

1. [Why This Project Exists](#1-why-this-project-exists)
2. [Hardware & Model Selection](#2-hardware--model-selection)
3. [Understanding LLMs — Parameters & Intelligence](#3-understanding-llms--parameters--intelligence)
4. [Chain of Thought — Trained vs Prompted](#4-chain-of-thought--trained-vs-prompted)
5. [Agent Types — All of Them](#5-agent-types--all-of-them)
6. [What We Are Building & Why](#6-what-we-are-building--why)
7. [Multi-Agent Architecture](#7-multi-agent-architecture)
8. [Model Capabilities & Honest Limitations](#8-model-capabilities--honest-limitations)
9. [Future Use Case: Trade Research Automation](#9-future-use-case-trade-research-automation)
10. [Fine-Tuning, RAG, and Knowledge Distillation — Later](#10-fine-tuning-rag-and-knowledge-distillation--later)
11. [Implementation Plan](#11-implementation-plan)
12. [Project Structure](#12-project-structure)
13. [Technology Stack](#13-technology-stack)

---

## 1. Why This Project Exists

This is a **learning project** with a practical end goal.

**The learning goal:** Understand how LLM agents work from the ground up — tool calling, chain-of-thought reasoning, ReAct loops, and multi-agent coordination. No frameworks hiding what is happening. Pure Python, every component transparent and under your control.

**The practical goal:** Build a working local multi-agent system that can later be tested against real use cases. The system is model-agnostic — we can swap between different LLMs (DeepSeek-R1, Llama 3.1, Qwen, etc.) via Ollama to find which works best for which tasks. Once we understand the practical limits of local models, we can decide what to build on top of them — including domain-specific tools like trade research automation.

**Why local and offline?**
- No data leaves your machine — full privacy
- No per-token API cost after hardware
- Full control over the model — can fine-tune on your own data later
- Works without internet (except tools that explicitly need it, like web search)

---

## 2. Hardware & Model Selection

### Your Hardware

| Component | Spec | Notes |
|---|---|---|
| OS | Windows 11 | All tools run natively on Windows |
| RAM | 32GB | Handles 14B models with partial CPU offload |
| GPU | NVIDIA RTX 4070 Laptop, 8GB VRAM | Fits 7-9B models fully on GPU |
| CPU | Latest | Handles overflow layers from 14B models |

### Inference Server: Ollama

We use **Ollama** to serve LLMs locally. Ollama handles model downloading, quantization, and serving behind an OpenAI-compatible API. It runs natively on Windows — no WSL or Docker needed.

Note: The project originally planned to use vLLM, but vLLM does not support Windows. Ollama provides the same OpenAI-compatible API with simpler setup.

### Default Model: DeepSeek-R1 8B

DeepSeek-R1 is a "thinking" model — it has **trained chain of thought** (see Section 4). It reasons through problems internally before answering, which makes it the strongest reasoning model at this size. This is our default for testing the agent system.

To change the model, edit `MODEL_NAME` in `.env`. The agent code does not change.

### Models We Test With

All models run via Ollama. The agent code is model-agnostic — swap models by changing one line in `.env`.

**Fits fully on 8GB VRAM (fast):**

| Model | Best For | Command |
|---|---|---|
| deepseek-r1:8b | Complex reasoning, chain-of-thought (default) | `ollama pull deepseek-r1:8b` |
| llama3.1:8b | Fast general chat, most reliable | `ollama pull llama3.1:8b` |
| qwen2.5-coder:7b | Code generation and fixing | `ollama pull qwen2.5-coder:7b` |
| gemma2:9b | Research, complex reasoning | `ollama pull gemma2:9b` |

**14B models (partial CPU offload, slower but smarter):**

| Model | Best For | Command |
|---|---|---|
| deepseek-r1:14b | Better reasoning than 8b | `ollama pull deepseek-r1:14b` |
| phi4:14b | High quality general + creative | `ollama pull phi4:14b` |
| qwen2.5-coder:14b | Stronger code than 7b | `ollama pull qwen2.5-coder:14b` |

14B models don't fully fit in 8GB VRAM. Ollama automatically offloads overflow layers to CPU RAM (you have 32GB, so plenty of room). This means slower token generation but noticeably better quality. Worth testing to see if the speed trade-off is acceptable.

### What "8B" and "14B" Mean

8B = 8 billion parameters. Not 8GB. It is a coincidence that 8B models quantized to Q4 need ~5.5GB and happen to fit on 8GB GPUs. 14B quantized needs ~8-9GB — tight on 8GB VRAM, hence the partial CPU offload.

### Can LLMs Use Shared GPU Memory (Intel iGPU)?

No. The "shared memory" in Windows Task Manager is CPU RAM reserved for the Intel integrated graphics. Ollama uses your NVIDIA GPU via CUDA exclusively. It cannot use the Intel iGPU. However, Ollama can offload model layers to regular CPU RAM when VRAM is full — this is how 14B models work on your hardware.

### Why Not Claude or GPT-4 API?

Claude and GPT-4 class models are significantly more capable for complex reasoning. However:

- Cost per token adds up quickly for an agent system that makes many calls
- Your data (prompts, tool results, code) leaves your machine
- You cannot fine-tune them on your own data
- They require internet

The architecture is designed so you can swap the model backend later. You can switch between any Ollama model by changing `.env`, or route specific agents to a cloud API if needed (e.g., orchestrator on Claude, specialists on local DeepSeek).

---

## 3. Understanding LLMs — Parameters & Intelligence

### What Parameters Are

Parameters are the learned connections (weights) inside a neural network — analogous to synapses in a brain. They are numbers adjusted during training to make the model better at predicting the next token.

DeepSeek-R1 8B has 8 billion such numbers. Frontier models like Claude are estimated at 200B+.

### What More Parameters Buy You

**Pattern storage capacity** — more parameters means the model memorized more relationships between concepts. A 9B model retains far fewer patterns than a 200B model from training.

**Reasoning depth** — larger models can hold more computational "context" internally while working through a problem. This is why bigger models handle multi-step reasoning more reliably.

**Generalization** — bigger models are better at applying knowledge to genuinely novel situations. Smaller models tend to be more rigid and pattern-matching.

**Nuance and calibration** — knowing when to say "it depends," handling ambiguity, and recognizing the limits of their own knowledge.

### Parameters Are Not Everything

| Factor | What It Affects |
|---|---|
| Parameter count | The ceiling of what is possible |
| Training data quality | What the model actually learned |
| Training method (RLHF, RL) | How well it follows instructions and reasons |
| Quantization | Reduces numerical precision, slightly reduces quality |
| Context window | How much the model can "see" at once |

A well-trained 9B model beats a poorly trained 50B model on specific tasks. A 9B model fine-tuned on your specific domain can outperform a general 200B model on that exact domain.

### The Simple Analogy

Parameters are like RAM. More RAM lets you run more complex programs. But RAM alone does not make software better — the quality of the code (training data and method) matters equally. A well-written program on 16GB RAM can outperform a bloated program on 64GB RAM for the right task.

---

## 4. Chain of Thought — Trained vs Prompted

This is a critical concept for this project. There are two fundamentally different ways an LLM can "think step by step."

### Trained Chain of Thought (CoT)

Some models are specifically trained to reason internally before producing an answer. Examples include OpenAI's o1 and DeepSeek-R2. During their training process, they were rewarded for showing their reasoning — so the ability to think step by step is **baked into the weights** of the model.

You cannot add trained CoT to most open-source models. However, **DeepSeek-R1 — our default model — does have trained CoT.** This is one reason we chose it. It reasons through problems internally before answering, which is a significant advantage over other models at this size.

### Prompted Chain of Thought (What We Build)

You design the reasoning structure yourself via prompts. You tell the model: "Before answering, think through the problem step by step in this format." The model follows your structure.

**ReAct** (Reasoning + Acting) is exactly this — we impose a `Thought → Act → Observe` pattern through the system prompt. The model did not learn this pattern during training. We are telling it to follow this pattern at inference time.

This is what we are implementing in this project.

### The Key Insight

**The chain is yours. The quality of reasoning within each step depends on the model's capacity.**

You can design a 10-step reasoning chain, but a 9B model might lose coherence by step 6 where a 200B model holds together through all 10. The chain structure helps the model stay organized, but it does not make the model smarter — it helps the model use its existing intelligence more consistently. With DeepSeek-R1, we get both: trained internal reasoning plus our prompted reasoning structure on top.

This is one of the things we want to test: where does each model start losing coherence in a prompted reasoning chain? And does DeepSeek-R1's trained CoT give it a meaningful advantage over other 8B models in our agent system?

---

## 5. Agent Types — All of Them

There are three main agent patterns, each building on the last. We describe all three here for completeness. **This project implements two of them: ReAct agents and Multi-Agent orchestration.**

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

**How it works:** The model receives a list of available tools (as JSON schemas) in its prompt. When it decides a tool is needed, it outputs a structured tool call (function name + arguments). Your code parses that output, executes the function, and feeds the result back to the model. The model then either calls another tool or produces a final answer.

**Best for:** Single focused tasks where the right tool is obvious.
**Example:** "What is 157 * 23?" → calls calculator tool → returns 3611.

**We are not using this as a standalone pattern** because it lacks explicit reasoning. The model jumps straight to tool calls without explaining why, which makes debugging hard and errors invisible.

---

### Type 2: ReAct Agent (Reasoning + Acting) ← We build this

Adds an explicit **Thought step** before every action. The model reasons out loud about what it knows, what it needs, and why it is choosing a particular tool — before acting.

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

**How it works:** The system prompt instructs the model to always output a `Thought` before deciding on an `Action`. This is **prompted chain of thought** — we design the reasoning format, the model fills it in. After each tool result (`Observe`), the model thinks again before proceeding.

**Why it is better than plain tool-calling:**
- Forcing explicit reasoning before each action catches mistakes early
- Prevents the model from jumping to conclusions
- Makes the agent's behavior fully debuggable — you can read the Thought steps and see exactly where it went wrong
- Gives the model a "scratchpad" to track what it has done and what is left

**Best for:** Multi-step tasks requiring sequential reasoning.
**Example:** "Find the latest unemployment data and calculate the year-over-year change."

**This is our core building block.** Every agent in this system uses a ReAct loop internally.

---

### Type 3: Multi-Agent System ← We build this

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

**How it works:** The Orchestrator receives a task, uses the LLM to break it into subtasks (as a JSON plan), then delegates each subtask to the appropriate specialist agent. Each specialist is itself a ReAct agent — so it reasons through its subtask step by step. When all specialists finish, the Orchestrator merges their results via one final LLM call.

**Why multi-agent genuinely helps (even with the same base model):**

- **Focused context per agent** — each agent's prompt is smaller and domain-specific. LLMs perform measurably better with focused, constrained prompts than open-ended complex ones
- **Separation of concerns** — a researcher is not distracted by coding concerns
- **Error catching via reviewer** — a fresh agent reviewing an output catches errors the original agent missed
- **Longer effective reasoning chain** — the chain of agents can reason across more steps than any single agent's context window allows
- **Parallel execution** — multiple agents can work simultaneously on different subtasks

**The honest limitation:** Multi-agent raises how consistently you operate near the model's intelligence ceiling, but does not raise the ceiling itself. Think of it as: one person doing 5 different jobs makes more mistakes than 5 people each doing 1 job — but they are all equally skilled. For tasks that require raw reasoning power beyond what the model can do, multi-agent will not fix that.

---

### Summary: What We Are Implementing

| Agent Type | Using It? | Role in This Project |
|---|---|---|
| Tool-Calling Agent | **No** (as standalone) | Subsumed by ReAct — every ReAct agent uses tool calling, but with reasoning added |
| ReAct Agent | **Yes** | The core building block. Every specialist agent uses a ReAct loop internally |
| Multi-Agent System | **Yes** | Orchestrator + specialists. Coordinates multiple ReAct agents |

---

## 6. What We Are Building & Why

### The Two Things We Want to Learn

**1. Prompted Chain of Thought (ReAct)**

Can we design a reasoning structure that makes local models more reliable? At what point does the model lose coherence in a multi-step chain? We build ReAct agents and test their limits — seeing where prompted CoT helps and where the model's parameter ceiling becomes the bottleneck.

**2. Multi-Agent Coordination**

Does splitting a complex task across multiple focused agents produce better results than one agent doing everything? We build an orchestrator and specialists to test this directly — same model, same hardware, but different architectural patterns.

### What We Want to Answer By The End

- Does ReAct (prompted CoT) measurably improve output quality over plain tool-calling for local models?
- Does DeepSeek-R1's trained CoT give it a meaningful edge over other 8B models in an agent system?
- Does multi-agent produce better results than a single ReAct agent for complex tasks?
- What types of tasks are local 7-14B models actually reliable for?
- Where do they consistently fail, and is that a prompting problem or a model capacity problem?
- Is this system useful enough to build real tools on top of (like trade research automation)?

---

## 7. Multi-Agent Architecture

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER TASK                             │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR                            │
│  - Receives task                                        │
│  - Produces JSON plan with subtasks                     │
│  - Delegates to specialists                             │
│  - Merges results into final answer                     │
└──────┬──────────────────┬──────────────────┬────────────┘
       ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│  RESEARCHER  │  │    CODER     │  │    REVIEWER       │
│              │  │              │  │                   │
│ Tools:       │  │ Tools:       │  │ Tools:            │
│ - web_search │  │ - run_code   │  │ - read_file       │
│ - read_file  │  │ - write_file │  │ - run_code        │
│              │  │ - read_file  │  │                   │
│ ReAct loop   │  │ ReAct loop   │  │ ReAct loop        │
└──────────────┘  └──────────────┘  └──────────────────┘
       ↓                  ↓                  ↓
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATOR MERGES RESULTS                 │
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
- Self-corrects on errors — runs code, reads error, fixes, runs again
- Returns working code + execution output to the orchestrator

**Reviewer Agent** *(Phase 4)*
- Receives the output from researcher or coder
- Checks for errors, logical gaps, or code bugs
- Returns a review verdict: PASS, or specific FEEDBACK to fix
- Creates a feedback loop — coder fixes based on reviewer notes

---

## 8. Model Capabilities & Honest Limitations

This section describes what to expect from local 7-14B parameter models in general. Specific strengths vary by model — DeepSeek-R1 is stronger at reasoning, Qwen is better at code, Llama is faster and more reliable for chat. Part of this project is testing these differences.

### What Local 7-14B Models Do Well

- Code generation for clear, scoped tasks (write a function, fix a bug, explain code)
- Following structured output instructions (JSON, markdown)
- Tool calling — understanding when to call tools and formatting arguments correctly
- Summarization and Q&A over provided documents
- Multi-turn conversation with memory
- **Small, well-defined tasks** — where instructions are clear and output format is constrained
- DeepSeek-R1 specifically: chain-of-thought reasoning (trained, not just prompted)

### Where They Struggle Compared to Frontier Models (Claude, GPT-4)

- **Complex multi-step reasoning** — more likely to lose track mid-chain (though DeepSeek-R1 is notably better here than other 8B models)
- **System design and architecture decisions** — tends toward generic answers
- **Math-heavy code** (ML algorithms, numerical methods) — inconsistent accuracy
- **Deep domain expertise** — no specialty, jack of all trades
- **Self-awareness of own errors** — less likely to catch own mistakes
- **Long reasoning chains** — a prompted 10-step chain may degrade after step 5-6

### The Bottom Line

Local 7-14B models are best suited for **specific, well-scoped, structured tasks** rather than broad general reasoning. Multi-agent helps them stay focused on such tasks. The combination of ReAct + multi-agent gets the most out of what these models can do — but it does not turn a small model into a frontier model.

Part of this project is discovering exactly where that line is, and whether different models draw that line in different places.

---

## 9. Future Use Case: Trade Research Automation

> **Status: Not finalized.** This use case will be tested and refined after the core multi-agent system is working and we have a clear understanding of the practical limits of local models.

### The Idea

Automate the processing of trade-related information — news articles, major announcements, and pre-scheduled economic events. The goal is not sentiment analysis but **structured automation**: knowing what events are happening, when, and converting unstructured text into structured, actionable data.

### What This Would Look Like

- **Input:** Unstructured text (news headlines, economic calendar entries, announcement text)
- **Output:** Structured data (event type, date/time, affected instruments, severity)
- **Interface:** Natural language commands instead of writing a script for each case — turning unstructured data into structured commands

### Why It Fits This Architecture

This is exactly the kind of task where a multi-agent system with a 9B model could work well:
- Each subtask is small and well-scoped (parse this text, classify this event, extract this date)
- The output format is structured and constrained
- It does not require deep open-ended reasoning

### Path Forward

Once the multi-agent system is tested, we will evaluate whether local models handle this use case reliably. If they do, the next step could be fine-tuning, RAG, or knowledge distillation to create a smaller specialized tool. That decision comes after testing, not before.

---

## 10. Fine-Tuning, RAG, and Knowledge Distillation — Later

These are three paths for making the system domain-specific. We document them here for reference. **None are being implemented now** — they depend on what we learn from testing the base system.

### Fine-Tuning

Takes the base model weights and further trains them on your specific domain data. Shifts the model's probability distribution toward your domain without erasing general knowledge.

**When it makes sense:** You have 500-1000+ high quality examples in a specific domain, and the base model is close but not accurate enough on that domain.

**Tools:** Unsloth + LoRA (memory efficient, works on 8GB GPU).

### RAG (Retrieval-Augmented Generation)

Instead of training new knowledge into the model, you store your domain data in a vector database and retrieve relevant chunks at query time. The model reasons over the retrieved context.

**When it makes sense:** You have a large corpus of documents the model needs to reference, and the information changes frequently.

**Tools:** ChromaDB or Qdrant (local, no cloud dependency).

### Knowledge Distillation

Use a larger model (or our best local model) as a "teacher" to generate labeled data, then train a much smaller, faster model (the "student") to mimic the teacher on a narrow task.

**When it makes sense:** You need a very fast, lightweight model for a specific task (like classifying news events in a trading pipeline) and even an 8B model is too slow or heavy for production use.

**Example path:** DeepSeek-R1 labels 10,000 news articles → train a small classifier on those labels → classifier runs in milliseconds in your pipeline.

---

## 11. Implementation Plan

### Phase 1: Foundation — Get the LLM Running

**Goal:** Ollama serving DeepSeek-R1 locally, a basic agent loop that can call tools and return answers.

| Step | What | Details |
|---|---|---|
| 1 | Environment setup | Install dependencies (`pip install requests python-dotenv`), create project structure |
| 2 | Download model | `ollama pull deepseek-r1:8b` — Ollama handles everything |
| 3 | Verify server | Run `python start_server.py` — checks Ollama is running and model is available |
| 4 | LLM client | `core/llm_client.py` — HTTP communication with Ollama's OpenAI-compatible API |
| 5 | Tool definitions | `core/tools.py` — implement tools, JSON schemas, executor dispatcher |
| 6 | Base agent class | `core/agent.py` — tool-calling loop, history management, max rounds safety |

**Done when:** A single agent calls two tools in sequence and returns a correct answer.

---

### Phase 2: ReAct — Prompted Chain of Thought

**Goal:** Every agent reasons explicitly (Thought → Act → Observe) before every action.

| Step | What | Details |
|---|---|---|
| 1 | ReAct system prompts | Add Thought/Act/Observe format instructions to agent prompts |
| 2 | Update agent loop | Parse and display Thought steps separately, store traces for debugging |
| 3 | Test and compare | Run same tasks with and without ReAct, compare quality and error rate |

**Done when:** Agent produces visible Thought → Act → Observe cycles. Errors are catchable by reading the thought trace.

---

### Phase 3: Multi-Agent Orchestration

**Goal:** Orchestrator delegates tasks to Researcher and Coder, merges results.

| Step | What | Details |
|---|---|---|
| 1 | Researcher agent | `agents/researcher.py` — information gathering, web_search + read_file |
| 2 | Coder agent | `agents/coder.py` — code writing/testing, run_code + write_file + read_file |
| 3 | Orchestrator | `agents/orchestrator.py` — JSON plan, delegation, result merging |
| 4 | Entry point | `run.py` — interactive loop, shows which agent is working |

**Done when:** A task requiring both research and coding is correctly split, delegated, and merged.

---

### Phase 4: Reviewer Agent + Feedback Loop

**Goal:** Output quality improves through self-correction cycles.

| Step | What | Details |
|---|---|---|
| 1 | Reviewer agent | `agents/reviewer.py` — finds errors and gaps, returns PASS or FEEDBACK |
| 2 | Feedback loop | Orchestrator routes feedback back to the relevant agent for correction |
| 3 | Evaluation | Build test set, measure pass rate with and without reviewer |

**Done when:** A task with a deliberate bug gets caught by the reviewer and corrected by the coder.

---

### Phase 5: Test with Real Use Case

**Goal:** Apply the system to a real task and evaluate practical limits of local models.

| Step | What | Details |
|---|---|---|
| 1 | Pick a use case | Trade research automation or another concrete task |
| 2 | Build task-specific tools | E.g., parse economic calendar, classify event type |
| 3 | Run and evaluate | Does it work reliably? Where does it fail? |
| 4 | Decide next step | Fine-tuning, RAG, distillation, or different model |

---

## 12. Project Structure

```
multi_agent/
│
├── README.md                    ← This document
├── .env                         ← Configuration (model name, Ollama URL, etc.)
├── requirements.txt             ← Python dependencies (minimal — requests, dotenv)
├── run.py                       ← Entry point — interactive loop
├── download_model.py            ← Pulls model via Ollama
├── start_server.py              ← Health check — verifies Ollama is running
│
├── core/
│   ├── llm_client.py            ← Ollama API communication
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
    ├── test_tools.py            ← Unit tests for each tool
    ├── test_agents.py           ← Integration tests per agent
    └── eval_tasks.json          ← Test tasks with expected outputs
```

---

## 13. Technology Stack

| Component | Technology | Why |
|---|---|---|
| LLM (default) | DeepSeek-R1 8B via Ollama | Best reasoning at this size, trained CoT, fits 8GB VRAM |
| LLMs (testing) | Llama 3.1, Qwen2.5-Coder, Gemma2, Phi-4 | Different strengths — swap via `.env` |
| Inference server | Ollama | Runs on Windows, OpenAI-compatible API, handles quantization |
| Agent framework | Pure Python | Full transparency, no framework lock-in |
| HTTP client | requests | Simple, no overhead |
| Code execution tool | subprocess | Built-in Python, no extra dependencies |
| Fine-tuning (later) | Unsloth + LoRA | Memory efficient, works on 8GB GPU |
| Vector DB for RAG (future) | ChromaDB or Qdrant | Local, no cloud dependency |

---

## Quick Reference — Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| Inference server | Ollama (not vLLM) | vLLM doesn't run on Windows. Ollama runs natively, same OpenAI-compatible API |
| Default model | DeepSeek-R1 8B | Best reasoning at this size, trained chain-of-thought |
| Single model vs multi | Multi-model testing | Different models excel at different tasks — swap via `.env` |
| Framework vs Pure Python | Pure Python | Full control, no abstraction hiding bugs |
| Agent patterns | ReAct + Multi-Agent | ReAct for reasoning, multi-agent for task coordination |
| Trained vs Prompted CoT | Both | DeepSeek-R1 has trained CoT; we add prompted CoT (ReAct) on top |
| Fine-tune vs RAG vs Distillation | Decide after testing | Need to understand model limits on real tasks first |
| Local vs API model | Local | Privacy, cost, and fine-tuning control |