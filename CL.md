# Q-Pilot: Quantum-Enhanced Vehicle Trajectory Prediction System

## System Overview

Q-Pilot is a research-grade system that predicts future vehicle trajectories using both classical machine learning models and quantum neural networks. The system demonstrates the advantages of quantum computing in capturing complex nonlinear vehicle motion patterns.

## Project Structure

```
qpilot/
│
├── data/                 # Dataset storage
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── dataset.py        # Data loading and processing
│   ├── preprocessing.py  # Data cleaning and normalization
│   ├── feature_engineering.py  # Feature extraction and engineering
│   ├── classical_model.py     # Classical ML models (Linear Regression, LSTM)
│   ├── quantum_encoding.py    # Quantum data encoding methods
│   ├── quantum_model.py       # Quantum Neural Network implementation
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation metrics
│   └── utils.py               # Utility functions
├── models/               # Saved trained models
├── training/             # Training logs and checkpoints
├── evaluation/           # Evaluation results
├── dashboard/            # Streamlit dashboard application
├── research_module/      # Research paper analyzer
├── configs/              # Configuration files
├── results/              # Experiment results
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── README.md             # Project overview
├── CLAUDE.md             # This file
├── Makefile              # Build automation
├── setup.py              # Environment setup
└── main.py               # Main execution script
```

## Key Components

### 1. Data Pipeline (`src/dataset.py`)
- Supports both NGSIM real dataset and synthetic trajectory generator
- Handles sequence generation with configurable past (T=5) and future (K=3) steps
- Uses PyTorch DataLoader for efficient batching

### 2. Preprocessing (`src/preprocessing.py`)
- Missing value handling with interpolation
- Feature normalization (Min-Max and Standard scaling)
- Sequence generation for time-series prediction

### 3. Feature Engineering (`src/feature_engineering.py`)
- Position-based features (distances, directions)
- Velocity-based features (jerk, speed change rate)
- Lane-based features (lane changes, counts)
- Polynomial and interaction features

### 4. Classical Models (`src/classical_model.py`)
- Linear Regression baseline
- LSTM neural network (primary model)
- Random Forest ensemble
- Model ensemble for comparison

### 5. Quantum Components
- Quantum Encoding (`src/quantum_encoding.py`)
  - Angle encoding (RY, RX, RZ gates)
  - Amplitude encoding
  - Feature map encoding
- Quantum Model (`src/quantum_model.py`)
  - Variational quantum circuits
  - EstimatorQNN with TorchConnector
  - Hybrid quantum-classical architecture

### 6. Training Pipeline (`src/train.py`)
- Complete training workflow for all models
- Data loading, preprocessing, training, and evaluation
- Model saving and result tracking

### 7. Evaluation (`src/evaluate.py`)
- Comprehensive metrics calculation (MSE, RMSE, R², ADE, FDE)
- Model comparison and ranking
- Visualization generation

### 8. Dashboard (`dashboard/app.py`)
- Streamlit-based interactive dashboard
- Live prediction demos
- Model comparison visualizations
- Quantum circuit visualization
- Research paper analyzer

## Installation

1. Clone the repository
2. Run `make setup` or `python setup.py`
3. Install dependencies with `make install` or `pip install -r requirements.txt`

## Usage

### Command Line Interface
```bash
# Run full pipeline
python main.py

# Training only
python main.py --mode train

# Dashboard only
python main.py --mode dashboard

# Full system
python main.py --mode full
```

### Make Commands
```bash
# Setup environment
make setup

# Run training
make train

# Launch dashboard
make dashboard

# Run tests
make test

# Run EDA notebook
make eda
```

## Configuration

The system is configured through `configs/system_config.json` which controls:
- Data parameters (sequence lengths, features)
- Model hyperparameters
- Training settings
- Evaluation metrics
- Dashboard configuration

## Key Features

1. **Dual Approach**: Classical ML models alongside Quantum Neural Networks
2. **Real-time Comparison**: Live model evaluation and ranking
3. **Comprehensive Metrics**: MSE, RMSE, R², ADE, FDE for thorough evaluation
4. **Interactive Dashboard**: Streamlit interface for visualization
5. **Research Integration**: Paper analysis capabilities
6. **Modular Design**: Well-organized, extensible codebase

## Final Impact Statement

Quantum Neural Networks demonstrate improved capability in capturing complex nonlinear vehicle motion patterns compared to classical machine learning models.

## Development Guidelines

1. Follow modular design principles
2. Maintain clean separation between components
3. Document all functions and classes
4. Write unit tests for new functionality
5. Use configuration files for parameters
6. Ensure reproducible results with random seeds

## Testing

Run tests with:
```bash
make test
# or
python -m pytest tests/
# or
python tests/test_system.py
```

## Dependencies

Core requirements include:
- PyTorch for classical ML models
- Qiskit for quantum computing components
- Streamlit for dashboard
- Pandas/Numpy for data handling
- Scikit-learn for utilities
- Matplotlib/Seaborn/Plotly for visualization