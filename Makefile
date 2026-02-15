PYTHON ?= python

.PHONY: install install-dev format lint typecheck test run ensure-indexes

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev]

format:
	$(PYTHON) -m black src tests
	$(PYTHON) -m ruff check src tests --fix

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

run-api:
	$(PYTHON) -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload --app-dir src

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m app $(ARGS)

ensure-indexes:
	$(PYTHON) -m app ensure-indexes
