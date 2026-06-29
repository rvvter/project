# Makefile
# One-command startup for the Learning Accelerator.
#
# Usage:
#   make setup      : create venv and install dependencies
#   make run        : run the main application
#   make langfuse   : start Langfuse observability stack
#   make clean      : remove generated files

.PHONY: setup run streamlit langfuse langfuse-stop langfuse-logs clean help

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# ── Setup ────────────────────────────────────────────────────────────────────

setup:
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@cp -n .env.example .env 2>/dev/null || true
	@mkdir -p data
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. source $(VENV)/bin/activate"
	@echo "  2. Edit .env, set OLLAMA_MODEL to match your VRAM"
	@echo "  3. ollama pull qwen2.5:7b  (or qwen2.5-coder:32b for 24GB)"
	@echo "  4. make run"

# ── Running ───────────────────────────────────────────────────────────────────

run:
	@echo "Starting Learning Accelerator..."
	$(PYTHON) main.py

streamlit:
	@echo "Starting Learning Accelerator (Streamlit UI)..."
	@echo "Open: http://localhost:8501"
	$(PYTHON) -m streamlit run streamlit_app.py

run-goal:
	@echo "Starting with custom goal: $(GOAL)"
	$(PYTHON) main.py "$(GOAL)"

resume:
	@echo "Resuming session: $(SESSION)"
	$(PYTHON) main.py --resume $(SESSION)

# ── Observability ─────────────────────────────────────────────────────────────

langfuse:
	@echo "Starting Langfuse..."
	docker compose up -d
	@echo ""
	@echo "Langfuse UI: http://localhost:3000"
	@echo "Create a project and add keys to .env"

langfuse-stop:
	docker compose down

langfuse-logs:
	docker compose logs -f langfuse-server

# ── Maintenance ───────────────────────────────────────────────────────────────

clean:
	@echo "Cleaning generated files..."
	@rm -f data/checkpoints.db
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete (venv preserved)"

clean-all: clean
	@rm -rf $(VENV)
	@echo "Full clean complete"

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Learning Accelerator, available commands:"
	@echo ""
	@echo "  make setup        Set up virtual environment and install deps"
	@echo "  make run          Run the main application"
	@echo "  make run-goal GOAL='Learn async Python'  Run with custom goal"
	@echo "  make resume SESSION=abc12345  Resume a stopped session"
	@echo ""
	@echo "  make langfuse     Start Langfuse observability (Docker required)"
	@echo "  make langfuse-stop  Stop Langfuse"
	@echo ""
	@echo "  make clean        Remove checkpoints and cache files"
