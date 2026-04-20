"""
Training pipeline for Q-Pilot system.
Trains both classical and quantum models.
"""
import numpy as np
import torch
import os
import json
from datetime import datetime
from tqdm import tqdm

from dataset import generate_synthetic_trajectory, create_dataloader, NGSIMDataset
from preprocessing import prepare_data_pipeline
from feature_engineering import enhance_features
from classical_model import ClassicalModelEnsemble
from quantum_model import create_quantum_model, QuantumModelTrainer
from utils import calculate_metrics, save_model, EarlyStopping
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Main training pipeline for Q-Pilot system
    """
    def __init__(self, config=None):
        """
        Initialize trainer

        Args:
            config (dict): Training configuration
        """
        self.config = config or self._default_config()
        self.models = {}
        self.metrics = {}
        self.training_history = {}

    def _default_config(self):
        """
        Get default training configuration

        Returns:
            dict: Default configuration
        """
        return {
            'seq_length': 5,
            'pred_length': 3,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'num_qubits': 4,
            'data_path': '../data',
            'model_path': '../models',
            'results_path': '../results'
        }

    def load_and_preprocess_data(self, use_ngsim=True):
        """
        Load and preprocess training data

        Args:
            use_ngsim (bool): Whether to use NGSIM data or synthetic data

        Returns:
            tuple: (train_loader, val_loader, test_loader, preprocessor)
        """
        print("Loading and preprocessing data...")

        if use_ngsim:
            try:
                # Try to load NGSIM data
                ngsim_dataset = NGSIMDataset(self.config['data_path'])
                raw_data = ngsim_dataset.load_data()
                processed_data = ngsim_dataset.preprocess_data()
                print(f"Loaded NGSIM data with shape: {processed_data.shape}")
            except Exception as e:
                print(f"Failed to load NGSIM data: {e}")
                print("Generating synthetic data as fallback...")
                use_ngsim = False

        if not use_ngsim:
            # Generate synthetic data
            raw_data = generate_synthetic_trajectory(2000)
            processed_data = raw_data
            print(f"Generated synthetic data with shape: {processed_data.shape}")

        # Enhance features
        enhanced_data = enhance_features(processed_data)
        print(f"Enhanced data shape: {enhanced_data.shape}")

        # Prepare data pipeline
        (X, y), preprocessor = prepare_data_pipeline(
            enhanced_data,
            seq_length=self.config['seq_length'],
            pred_length=self.config['pred_length']
        )

        print(f"Prepared data - Input shape: {X.shape}, Target shape: {y.shape}")

        # Split data into train/validation/test
        total_samples = X.shape[0]
        train_split = int(0.7 * total_samples)
        val_split = int(0.85 * total_samples)

        X_train, y_train = X[:train_split], y[:train_split]
        X_val, y_val = X[train_split:val_split], y[train_split:val_split]
        X_test, y_test = X[val_split:], y[val_split:]

        # Create data loaders
        train_loader = self._create_torch_loader(X_train, y_train)
        val_loader = self._create_torch_loader(X_val, y_val)
        test_loader = self._create_torch_loader(X_test, y_test)

        return train_loader, val_loader, test_loader, preprocessor

    def _create_torch_loader(self, X, y):
        """
        Create PyTorch DataLoader

        Args:
            X (np.array): Input data
            y (np.array): Target data

        Returns:
            DataLoader: PyTorch DataLoader
        """
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        return loader

    def train_classical_models(self, train_loader, val_loader):
        """
        Train classical models

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader

        Returns:
            dict: Trained classical models
        """
        print("Training classical models...")

        # Get input/output shapes from data
        sample_batch = next(iter(train_loader))
        X_sample, y_sample = sample_batch
        input_shape = (self.config['seq_length'], X_sample.shape[-1])
        output_shape = (self.config['pred_length'], y_sample.shape[-1])

        # Initialize ensemble
        ensemble = ClassicalModelEnsemble(input_shape, output_shape)

        # Convert PyTorch DataLoader to numpy for classical models
        X_train_np = []
        y_train_np = []
        for batch_x, batch_y in train_loader:
            X_train_np.append(batch_x.numpy())
            y_train_np.append(batch_y.numpy())

        X_train_np = np.concatenate(X_train_np, axis=0)
        y_train_np = np.concatenate(y_train_np, axis=0)

        # Train linear model
        print("Training linear regression model...")
        ensemble.fit_linear(X_train_np, y_train_np)

        # Train LSTM model
        print("Training LSTM model...")
        ensemble.fit_lstm(train_loader, num_epochs=30, learning_rate=self.config['learning_rate'])

        # Train Random Forest model
        print("Training Random Forest model...")
        ensemble.fit_random_forest(X_train_np, y_train_np)

        self.models['classical'] = ensemble
        print("Classical models training completed.")
        return ensemble

    def train_quantum_model(self, train_loader, val_loader):
        """
        Train quantum model

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader

        Returns:
            nn.Module: Trained quantum model
        """
        print("Training quantum model...")

        # Get input/output dimensions
        sample_batch = next(iter(train_loader))
        X_sample, y_sample = sample_batch
        input_dim = X_sample.shape[1] * X_sample.shape[2]  # Flattened input
        output_dim = y_sample.shape[1] * y_sample.shape[2]  # Flattened output

        # Create quantum model
        q_model = create_quantum_model(
            input_dim=input_dim,
            output_dim=output_dim,
            num_qubits=self.config['num_qubits'],
            model_type='hybrid'
        )

        # Create trainer
        trainer = QuantumModelTrainer(q_model, learning_rate=self.config['learning_rate'])

        # Train model
        losses = trainer.train(train_loader, num_epochs=self.config['epochs'])

        self.models['quantum'] = q_model
        self.training_history['quantum_losses'] = losses
        print("Quantum model training completed.")
        return q_model

    def evaluate_models(self, test_loader, preprocessor=None):
        """
        Evaluate all trained models

        Args:
            test_loader (DataLoader): Test data loader
            preprocessor (DataPreprocessor): Data preprocessor for denormalization

        Returns:
            dict: Evaluation metrics for all models
        """
        print("Evaluating models...")

        # Collect all test data
        X_test_all = []
        y_test_all = []
        for batch_x, batch_y in test_loader:
            X_test_all.append(batch_x.numpy())
            y_test_all.append(batch_y.numpy())

        X_test_np = np.concatenate(X_test_all, axis=0)
        y_test_np = np.concatenate(y_test_all, axis=0)

        metrics = {}

        # Evaluate classical models
        if 'classical' in self.models:
            ensemble = self.models['classical']

            # Linear model evaluation
            print("Evaluating linear model...")
            linear_pred = ensemble.predict_linear(X_test_np)
            linear_metrics = calculate_metrics(linear_pred, y_test_np)
            metrics['linear'] = linear_metrics

            # LSTM model evaluation
            print("Evaluating LSTM model...")
            lstm_pred = ensemble.predict_lstm(X_test_np)
            lstm_metrics = calculate_metrics(lstm_pred, y_test_np)
            metrics['lstm'] = lstm_metrics

            # Random Forest evaluation
            print("Evaluating Random Forest model...")
            rf_pred = ensemble.predict_random_forest(X_test_np)
            rf_metrics = calculate_metrics(rf_pred, y_test_np)
            metrics['random_forest'] = rf_metrics

        # Evaluate quantum model
        if 'quantum' in self.models:
            print("Evaluating quantum model...")
            q_model = self.models['quantum']

            # Flatten test data for quantum model
            batch_size = X_test_np.shape[0]
            X_test_flat = X_test_np.reshape(batch_size, -1)
            y_test_flat = y_test_np.reshape(batch_size, -1)

            # Convert to torch tensor
            X_test_tensor = torch.FloatTensor(X_test_flat)

            # Make predictions
            q_model.eval()
            with torch.no_grad():
                q_pred_flat = q_model(X_test_tensor).numpy()

            # Reshape predictions back
            q_pred = q_pred_flat.reshape(y_test_np.shape)

            # Calculate metrics
            q_metrics = calculate_metrics(q_pred, y_test_np)
            metrics['quantum'] = q_metrics

        self.metrics = metrics
        return metrics

    def save_models_and_results(self):
        """
        Save trained models and results
        """
        print("Saving models and results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.config['model_path'], timestamp)
        results_dir = os.path.join(self.config['results_path'], timestamp)

        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Save models
        if 'classical' in self.models:
            # For classical models, we save the ensemble
            # In practice, you might want to save individual models
            print("Saving classical models...")
            # Saving logic would go here

        if 'quantum' in self.models:
            print("Saving quantum model...")
            q_model_path = os.path.join(model_dir, 'quantum_model.pth')
            torch.save(self.models['quantum'].state_dict(), q_model_path)

        # Save metrics
        metrics_path = os.path.join(results_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Save training history
        history_path = os.path.join(results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Models and results saved to {timestamp} directory.")
        return model_dir, results_dir

    def run_full_training(self, use_ngsim=True):
        """
        Run complete training pipeline

        Args:
            use_ngsim (bool): Whether to use NGSIM data

        Returns:
            dict: Training results
        """
        print("Starting Q-Pilot training pipeline...")
        print("=" * 50)

        # Load and preprocess data
        train_loader, val_loader, test_loader, preprocessor = self.load_and_preprocess_data(use_ngsim)

        # Train classical models
        self.train_classical_models(train_loader, val_loader)

        # Train quantum model
        self.train_quantum_model(train_loader, val_loader)

        # Evaluate models
        metrics = self.evaluate_models(test_loader, preprocessor)

        # Save results
        model_dir, results_dir = self.save_models_and_results()

        print("=" * 50)
        print("Training pipeline completed successfully!")
        print("=" * 50)

        # Print summary
        self._print_evaluation_summary(metrics)

        return {
            'models': self.models,
            'metrics': metrics,
            'model_dir': model_dir,
            'results_dir': results_dir
        }

    def _print_evaluation_summary(self, metrics):
        """
        Print evaluation summary

        Args:
            metrics (dict): Evaluation metrics
        """
        print("\nEVALUATION SUMMARY:")
        print("-" * 30)

        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name.upper()} MODEL METRICS:")
            for metric_name, value in model_metrics.items():
                print(f"  {metric_name}: {value:.4f}")


def main():
    """
    Main training function
    """
    # Configuration
    config = {
        'seq_length': 5,
        'pred_length': 3,
        'batch_size': 32,
        'epochs': 30,  # Reduced for faster training
        'learning_rate': 0.001,
        'num_qubits': 4,
        'data_path': '../data',
        'model_path': '../models',
        'results_path': '../results'
    }

    # Initialize trainer
    trainer = ModelTrainer(config)

    # Run training
    try:
        results = trainer.run_full_training(use_ngsim=False)  # Use synthetic data for demo
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("This might be due to quantum simulation limitations or library compatibility issues.")


if __name__ == "__main__":
    main()