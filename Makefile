.PHONY: install run-batch lint help

help:
	@echo "Fraud Service - Available targets:"
	@echo "  make install      - Install package in editable mode"
	@echo "  make run-batch    - Score transactions and write results"
	@echo "  make lint         - Run ruff check on src/ and tests/"

install:
	python -m pip install -e .

run-batch:
	python -m fraud_service.batch

lint:
	ruff check src tests 2>/dev/null || ruff check src
