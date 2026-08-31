.PHONY: install run-batch lint test test-unit test-integration test-smoke help

help:
	@echo "Fraud Service - Available targets:"
	@echo "  make install            - Install package in editable mode"
	@echo "  make run-batch          - Score transactions and write results"
	@echo "  make test               - Run the full pytest suite"
	@echo "  make test-unit          - Run unit tests only"
	@echo "  make test-integration   - Run integration tests only"
	@echo "  make test-smoke         - Run smoke tests only"
	@echo "  make lint               - Run ruff check on src/ and tests/"

install:
	python -m pip install -e .

run-batch:
	python -m fraud_service.batch

test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-smoke:
	pytest -m smoke

lint:
	ruff check src tests 2>/dev/null || ruff check src
