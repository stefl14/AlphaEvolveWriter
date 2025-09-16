.PHONY: help install run test clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make run      - Run the API server"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean cache files"

install:
	pip install -e .

run:
	uvicorn kraken.api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete