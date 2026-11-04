PY      ?= python3.12
VENV    := .venv
BIN     := $(VENV)/bin
PORT    ?= 8000

.PHONY: install install-all demo sample run backtest clean

install:            ## core + local-model deps (enough for the offline demo)
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install --prefer-binary -e ".[ollama]"

install-all:        ## everything: Ollama, Gemini, live data, tracing
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install --prefer-binary -e ".[all]"

demo:               ## serve the UI with the bundled example runs
	@echo "Open http://localhost:$(PORT)"
	STRICT_MODEL=0 $(BIN)/uvicorn agentic_bl.server.app:app --port $(PORT)

run:                ## one offline rebalance (fixture data + mock agents)
	STRICT_MODEL=0 $(BIN)/agentic-bl-run

backtest:           ## short offline backtest
	STRICT_MODEL=0 $(BIN)/agentic-bl-backtest --start 2024-01-31 --end 2024-06-30

sample:             ## regenerate examples/ from a fresh offline run + backtest
	$(BIN)/python scripts/make_examples.py

clean:
	rm -rf runs .cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
