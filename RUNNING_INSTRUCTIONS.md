# Running Q-Pilot System

## Prerequisites

Before running Q-Pilot, ensure you have the following installed:

1. **Python 3.7+** - Download from [python.org](https://www.python.org/downloads/)
2. **pip** - Python package manager (usually included with Python)

## Installation Steps

### 1. Navigate to the Q-Pilot Directory
```bash
cd qpilot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install all required packages:
- torch>=1.12.0
- torchvision>=0.13.0
- qiskit>=0.40.0
- qiskit-machine-learning>=0.5.0
- streamlit>=1.18.0
- pandas>=1.5.0
- numpy>=1.21.0
- scikit-learn>=1.1.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- plotly>=5.11.0
- tqdm>=4.64.0
- PyPDF2>=3.0.0

### 3. Run Setup Script (Optional)
```bash
python setup.py
```

## Running the System

### Option 1: Quick Demo
```bash
python run_demo.py
```

### Option 2: Full Demo
```bash
python run_demo.py --full
```

### Option 3: Interactive Mode
```bash
python main.py
```

This will present you with options:
1. Training pipeline
2. Interactive dashboard
3. Full system (training + dashboard)

### Option 4: Training Only
```bash
python main.py --mode train
```

### Option 5: Dashboard Only
```bash
python main.py --mode dashboard
```

### Option 6: Using Make Commands (if Make is available)
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

## Using the Dashboard

After launching the dashboard with `python main.py --mode dashboard` or `make dashboard`:

1. Open your web browser to `http://localhost:8501`
2. Explore the tabs:
   - **Live Prediction Demo**: See models in action
   - **Model Comparison**: Compare performance metrics
   - **Quantum Circuit Visualization**: View quantum circuits
   - **Dataset Explorer**: Analyze trajectory data
   - **Research Paper Analyzer**: Analyze academic papers

## Running Jupyter Notebooks

To explore the notebooks:

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open notebooks in the `notebooks/` directory:
   - `eda.ipynb`: Exploratory Data Analysis
   - `model_comparison.ipynb`: Model comparison analysis

## System Components

### Data Pipeline
- Supports both NGSIM dataset and synthetic trajectory generator
- Sequence length: 5 past steps
- Prediction length: 3 future steps

### Classical Models
- Linear Regression (baseline)
- LSTM Neural Network (primary)
- Random Forest (ensemble)

### Quantum Model
- 4-qubit variational quantum circuit
- Angle encoding with RY gates
- CNOT entanglement layers
- Hybrid quantum-classical training

### Evaluation Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)

## Troubleshooting

### Python Not Found
If you get "Python was not found" error:
1. Install Python from [python.org](https://www.python.org/downloads/)
2. Add Python to your PATH during installation
3. Restart your command prompt

### Missing Dependencies
If you get import errors:
```bash
pip install -r requirements.txt
```

### Quantum Simulation Issues
If quantum components fail:
1. Ensure qiskit is properly installed
2. Check quantum hardware requirements
3. Consider using quantum simulators for development

## Final Impact Statement

Quantum Neural Networks demonstrate improved capability in capturing complex nonlinear vehicle motion patterns compared to classical machine learning models.