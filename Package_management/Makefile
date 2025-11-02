.PHONY: help install test run examples clean lint format setup

help:
	@echo "RAG Evaluation System - Available Commands"
	@echo "==========================================="
	@echo "make setup      - Complete setup (venv + install + env)"
	@echo "make install    - Install dependencies"
	@echo "make test       - Run tests"
	@echo "make run        - Run main evaluation"
	@echo "make examples   - Run example scripts"
	@echo "make lint       - Run code linters"
	@echo "make format     - Format code with black"
	@echo "make clean      - Clean temporary files"
	@echo "make docs       - Generate documentation"

setup:
	@echo "Setting up RAG Evaluation System..."
	python -m venv venv
	@echo "Activating virtual environment..."
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template"; \
		echo "Please edit .env and add your TOGETHER_API_KEY"; \
	fi
	@echo "Setup complete! Activate venv with: source venv/bin/activate"

install:
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy ipython jupyter
	@echo "Development dependencies installed!"

test:
	@echo "Running tests..."
	pytest test_evaluator.py -v --tb=short

test-cov:
	@echo "Running tests with coverage..."
	pytest test_evaluator.py -v --cov=evaluator --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

run:
	@echo "Running RAG evaluation..."
	python evaluator.py

examples:
	@echo "Running example scripts..."
	python example_usage.py

lint:
	@echo "Running linters..."
	flake8 evaluator.py config.py utils.py example_usage.py --max-line-length=100
	@echo "Linting complete!"

format:
	@echo "Formatting code with black..."
	black evaluator.py config.py utils.py example_usage.py test_evaluator.py --line-length=100
	@echo "Code formatted!"

type-check:
	@echo "Running type checker..."
	mypy evaluator.py config.py utils.py --ignore-missing-imports

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	@echo "Cleanup complete!"

clean-results:
	@echo "Cleaning result files..."
	rm -rf results/*.csv results/*.json results/*.html
	@echo "Results cleaned!"

docs:
	@echo "Documentation available in README.md and QUICKSTART.md"
	@echo "To view in browser, use: python -m http.server 8000"

check: lint type-check test
	@echo "All checks passed!"

package:
	@echo "Building package..."
	python setup.py sdist bdist_wheel
	@echo "Package built in dist/"

phoenix:
	@echo "Starting Phoenix UI..."
	@echo "Open http://localhost:6006 in your browser"
	python -c "import phoenix as px; session = px.launch_app(); print(f'Phoenix running at {session.url}'); import time; time.sleep(3600)"

demo:
	@echo "Running quick demo..."
	python -c "from example_usage import example_1_basic_evaluation; example_1_basic_evaluation()"

validate-env:
	@echo "Validating environment configuration..."
	@python -c "import os; from dotenv import load_dotenv; load_dotenv(); \
		key = os.getenv('TOGETHER_API_KEY'); \
		print('✓ TOGETHER_API_KEY is set' if key else '✗ TOGETHER_API_KEY not found in .env'); \
		exit(0 if key else 1)"

info:
	@echo "System Information"
	@echo "=================="
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "pip version:"
	@pip --version
	@echo ""
	@echo "Installed packages:"
	@pip list | grep -E "langchain|phoenix|together|pandas|numpy|scikit"

all: format lint test
	@echo "Full check complete!"