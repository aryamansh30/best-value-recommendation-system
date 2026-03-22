.PHONY: setup test test-unit test-live eval recommend api

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Runs all tests. Live API tests auto-skip if network/dependency unavailable.
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

# Deterministic/local tests only (no live FakeStore dependency).
test-unit:
	PYTHONPATH=src $(PYTHON) -m unittest \
		tests.test_query_parser \
		tests.test_retrieval \
		tests.test_ranking \
		tests.test_genai \
		tests.test_pipeline_fallback -v

# Live integration tests against FakeStore.
test-live:
	PYTHONPATH=src $(PYTHON) -m unittest tests.test_pipeline -v

eval:
	PYTHONPATH=src $(PYTHON) -m app.evaluation

recommend:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make recommend QUERY='Best electronics under $$150'"; \
		exit 1; \
	fi
	PYTHONPATH=src $(PYTHON) -m app.recommend --query "$(QUERY)" --top-k 5

api:
	PYTHONPATH=src $(VENV)/bin/uvicorn app.api:app --reload
