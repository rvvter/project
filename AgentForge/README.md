# Production-Grade Multi-Agent AI System

Build a complete multi-agent AI system that runs entirely on your machine.
No API keys, no cloud dependencies, no ongoing cost.

This project is the companion code for the freeCodeCamp handbook:
**How to Build a Local Multi-Agent AI System with LangGraph, MCP, and A2A**
(link to be added once published)

---

## What this builds

A **Learning Accelerator**: a four-agent system that plans a study curriculum,
explains topics from your own notes, quizzes you, and adapts based on results.
The use case is the teaching vehicle. The architecture is the real subject.

```
Goal: "Learn Python closures and decorators"
  │
  ▼
Curriculum Planner  →  structured study roadmap
  │
  ▼ (you approve the plan)
Explainer           →  reads your notes via MCP, explains each topic
  │
  ▼
Quiz Generator      →  tests understanding, grades answers with LLM-as-judge
  │
  ▼
Progress Coach      →  adapts roadmap, calls CrewAI Study Buddy via A2A
  │
  └── loops back to Explainer for next topic
```

The same architecture pattern runs in production for sales enablement,
compliance training, customer support onboarding, and engineering ramp-up.

---

## Architecture

| Layer | Technology | What it does |
|---|---|---|
| Orchestration | LangGraph 1.1.0 | Stateful agent graph with checkpointing |
| Tool integration | MCP (mcp 1.26.0) | Standardised agent-to-tool protocol |
| Agent coordination | A2A (a2a-sdk 0.3.25) | Cross-framework agent-to-agent protocol |
| Local inference | Ollama | LLM serving at localhost:11434 |
| Cross-framework | CrewAI 1.13.0 | Study Buddy agent (called via A2A) |
| Observability | Langfuse 4.0.1 | Full trace of every agent and LLM call |
| Evaluation | DeepEval 3.9.1 | LLM-as-judge quality metrics |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a full deep-dive.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Docker Desktop (for Langfuse observability, optional)
- 16 GB RAM minimum, 32 GB recommended
- 8 GB VRAM for `qwen2.5:7b` | 24 GB VRAM for `qwen2.5-coder:32b`

---

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/sandeepmb/freecodecamp-multi-agent-ai-system
cd freecodecamp-multi-agent-ai-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pull your model (choose based on VRAM)
ollama pull qwen2.5:7b          # 8 GB VRAM
ollama pull qwen2.5-coder:32b   # 24 GB VRAM

# 3. Ensure Ollama is running
ollama serve                  # in a separate terminal
# Or on macOS: start the Ollama app from your Applications folder

# 4. Configure
cp .env.example .env
# Edit .env: set OLLAMA_MODEL to match what you pulled

# 5. Run
python main.py
```

For the full system with A2A services and observability, see [Running all services](#running-all-services).

---

## Project structure

```
freecodecamp-multi-agent-ai-system/
├── src/
│   ├── agents/                 # LangGraph agent nodes
│   │   ├── curriculum_planner.py
│   │   ├── explainer.py
│   │   ├── quiz_generator.py
│   │   ├── progress_coach.py
│   │   └── human_approval.py
│   ├── graph/
│   │   ├── state.py            # Shared AgentState TypedDict
│   │   └── workflow.py         # LangGraph graph definition
│   ├── mcp_servers/            # MCP tool servers
│   │   ├── filesystem_server.py
│   │   └── memory_server.py
│   ├── a2a_services/           # A2A protocol services and client
│   │   ├── quiz_service.py     # Quiz Generator as A2A service
│   │   └── a2a_client.py       # Client for calling A2A services
│   ├── crewai_agent/
│   │   └── study_buddy.py      # CrewAI agent served via A2A
│   └── observability/
│       └── langfuse_setup.py   # Langfuse callback handler
├── tests/
│   ├── conftest.py             # Shared fixtures and markers
│   ├── test_state.py           # 24 tests
│   ├── test_curriculum_planner.py  # 11 tests
│   ├── test_mcp_servers.py     # 36 tests
│   ├── test_explainer.py       # 14 tests
│   ├── test_quiz_and_coach.py  # 17 tests
│   ├── test_checkpointing.py   # 20 tests
│   ├── test_observability.py   # 16 tests
│   ├── test_a2a.py             # 19 tests
│   ├── test_crewai_interop.py  # 25 tests
│   └── test_eval.py            # 12 eval tests (requires Ollama)
├── study_materials/
│   └── sample_notes/           # Markdown files the agents read
├── docs/
│   ├── ARCHITECTURE.md
│   └── MODEL_SELECTION.md
├── data/                       # SQLite checkpoint DB (created at runtime)
├── main.py                     # Entry point
├── docker-compose.yml          # Langfuse self-hosted stack
├── Makefile                    # One-command startup
├── requirements.txt
└── .env.example
```

---

## Running all services

The full system has three processes. Open three terminal tabs.

**Tab 1: Quiz Generator A2A service:**
```bash
source .venv/bin/activate
python src/a2a_services/quiz_service.py
# Serves at http://localhost:9001
```

**Tab 2: CrewAI Study Buddy A2A service:**
```bash
source .venv/bin/activate
python src/crewai_agent/study_buddy.py
# Serves at http://localhost:9002
```

**Tab 3: Main LangGraph application:**
```bash
source .venv/bin/activate
python main.py
```

Or use the Makefile:
```bash
make services   # starts both A2A services in background
make run        # runs the main application
```

---

## Session resume

Every session is checkpointed to `data/checkpoints.db` after each agent node.
To resume a stopped session:

```bash
python main.py --resume <session-id>
```

The session ID is printed at the start of every run.

---

## Observability

Start Langfuse locally:
```bash
docker compose up -d
# Open http://localhost:3000
```

Add your API keys to `.env` (from the Langfuse project settings), then run
as normal. Every agent call, LLM completion, and tool call appears in the
trace UI automatically.

---

## Testing

```bash
# Fast unit tests, run during development (~3 seconds)
# 182 tests across 9 test files
pytest tests/ -m "not eval" -v

# Quality evaluation tests, run before releases (~90 seconds, requires Ollama)
# 12 LLM-as-judge tests
pytest tests/test_eval.py -v -s -m eval
```

---

## Adding your own study materials

Replace or add Markdown files in `study_materials/sample_notes/`.
The Explainer agent reads every `.md` file in that directory automatically
via the MCP filesystem server. No configuration changes needed.

---

## Configuration reference

See `.env.example` for all available settings.

Key toggles:
| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model for all agents |
| `USE_A2A_QUIZ` | `true` | Route quiz tasks to A2A service |
| `USE_STUDY_BUDDY` | `true` | Call CrewAI Study Buddy for low scores |
| `CHECKPOINT_DB` | `data/checkpoints.db` | SQLite path for checkpoints |

---

## Article

This code is the companion to the freeCodeCamp handbook:
**How to Build Production-Grade Multi-Agent AI Systems with LangGraph, MCP, A2A, and Ollama**

The handbook explains every architectural decision, walks through each
agent step by step, and covers the production patterns that make this
system different from a basic tutorial.

---

## Author

**Sandeep Bharadwaj Mannapur**
Lead Data and AI/ML Engineer with 15+ years in financial services and enterprise SaaS.

[GitHub](https://github.com/sandeepmb) · [freeCodeCamp](https://www.freecodecamp.org/news/author/sandeep-mannapur/)
