# Makefile for Q-Pilot system

# Variables
PYTHON := python
PIP := pip
CONFIG := configs/system_config.json

# Default target
.PHONY: help
help:
	@echo "Q-Pilot Makefile"
	@echo "================="
	@echo "Available targets:"
	@echo "  setup     - Setup environment and install dependencies"
	@echo "  train     - Run training pipeline"
	@echo "  dashboard - Launch Streamlit dashboard"
	@echo "  full      - Run complete pipeline (train + dashboard)"
	@echo "  test      - Run system tests"
	@echo "  clean     - Clean temporary files"
	@echo "  notebook  - Start Jupyter notebook"
	@echo "  eda       - Run exploratory data analysis notebook"
	@echo "  compare   - Run model comparison notebook"

# Setup environment
.PHONY: setup
setup:
	$(PYTHON) setup.py

# Install dependencies only
.PHONY: install
install:
	$(PIP) install -r requirements.txt

# Run training pipeline
.PHONY: train
train:
	$(PYTHON) main.py --mode train --config $(CONFIG)

# Launch dashboard
.PHONY: dashboard
dashboard:
	$(PYTHON) main.py --mode dashboard

# Run full pipeline
.PHONY: full
full:
	$(PYTHON) main.py --mode full

# Run tests
.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

# Run unit tests
.PHONY: test-unit
test-unit:
	$(PYTHON) tests/test_system.py

# Clean temporary files
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/

# Start Jupyter notebook
.PHONY: notebook
notebook:
	jupyter notebook

# Run EDA notebook
.PHONY: eda
eda:
	jupyter notebook notebooks/eda.ipynb

# Run model comparison notebook
.PHONY: compare
compare:
	jupyter notebook notebooks/model_comparison.ipynb

# Create directories
.PHONY: dirs
dirs:
	mkdir -p data models training evaluation results notebooks configs dashboard research_module tests

# Print system info
.PHONY: info
info:
	@echo "System Information:"
	@echo "Python version: $$(python --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Available notebooks: $$(ls notebooks/*.ipynb 2>/dev/null | wc -l)"
	@echo "Configuration file: $(CONFIG)"