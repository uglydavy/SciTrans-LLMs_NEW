.PHONY: help install test lint format clean gui docs

help:
	@echo "SciTrans-LLMs Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linters"
	@echo "  format     - Format code"
	@echo "  clean      - Clean cache and temp files"
	@echo "  gui        - Launch GUI"
	@echo "  docs       - Generate documentation"
	@echo "  thesis     - Generate thesis materials"

install:
	pip install -r requirements-minimal.txt
	@echo "✓ Minimal dependencies installed"
	@echo "  To install ML packages: make install-ml"
	@echo "  To install dev tools: make install-dev"

install-ml:
	pip install -r requirements-ml.txt
	@echo "✓ ML dependencies installed"

install-dev:
	pip install -r requirements-dev.txt
	@echo "✓ Development tools installed"

install-all: install install-ml install-dev
	@echo "✓ All dependencies installed"

test:
	pytest tests/unit -v
	@echo "✓ Unit tests passed"

test-integration:
	pytest tests/integration -v --run-integration
	@echo "✓ Integration tests passed"

test-all:
	pytest tests/ -v --cov=scitran --cov-report=html
	@echo "✓ All tests completed. Coverage report in htmlcov/"

lint:
	flake8 scitran/ --max-line-length=100 --ignore=E203,W503
	mypy scitran/ --ignore-missing-imports
	@echo "✓ Linting passed"

format:
	black scitran/ tests/ gui/ --line-length=100
	isort scitran/ tests/ gui/ --profile black
	@echo "✓ Code formatted"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .cache htmlcov/
	@echo "✓ Cleaned cache files"

gui:
	python gui/app.py

quickstart:
	python quickstart.py

benchmark-speed:
	python benchmarks/speed_test.py

benchmark-quality:
	python benchmarks/quality_test.py corpus/test

ablation:
	python experiments/ablation.py corpus/test

thesis:
	@echo "Generating thesis materials..."
	python experiments/ablation.py corpus/test
	@echo "✓ Thesis materials generated in results/"

setup:
	./setup.sh

.DEFAULT_GOAL := help
