"""
Evaluation module for Q-Pilot system.
Comprehensive evaluation and comparison of classical and quantum models.
"""
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

from src.utils import calculate_metrics, calculate_ade, calculate_fde, calculate_rmse


class ModelEvaluator:
    """
    Comprehensive evaluator for trajectory prediction models
    """
    def __init__(self):
        self.results = {}
        self.comparisons = {}

    def evaluate_single_model(self, model, test_loader, model_name, preprocessor=None):
        """
        Evaluate a single model

        Args:
            model: Trained model
            test_loader: Test data loader
            model_name (str): Name of the model
            preprocessor: Data preprocessor for denormalization

        Returns:
            dict: Evaluation metrics
        """
        print(f"Evaluating {model_name} model...")

        # Collect all test data
        X_test_all = []
        y_test_all = []
        for batch_x, batch_y in test_loader:
            X_test_all.append(batch_x.numpy())
            y_test_all.append(batch_y.numpy())

        X_test_np = np.concatenate(X_test_all, axis=0)
        y_test_np = np.concatenate(y_test_all, axis=0)

        # Make predictions based on model type
        if model_name in ['linear', 'random_forest']:
            # Classical models that work with numpy
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test_np)
            elif hasattr(model, 'predict_linear'):
                predictions = model.predict_linear(X_test_np)
            elif hasattr(model, 'predict_random_forest'):
                predictions = model.predict_random_forest(X_test_np)
            else:
                raise ValueError(f"Unknown prediction method for {model_name}")
        else:
            # Models that work with PyTorch
            # Flatten input for quantum model
            batch_size = X_test_np.shape[0]
            X_test_flat = X_test_np.reshape(batch_size, -1)
            X_test_tensor = torch.FloatTensor(X_test_flat)

            model.eval()
            with torch.no_grad():
                pred_flat = model(X_test_tensor).numpy()
                predictions = pred_flat.reshape(y_test_np.shape)

        # Calculate metrics
        metrics = calculate_metrics(predictions, y_test_np)

        # Store results
        self.results[model_name] = {
            'predictions': predictions,
            'targets': y_test_np,
            'metrics': metrics
        }

        return metrics

    def compare_models(self, models_dict, test_loader, preprocessor=None):
        """
        Compare multiple models

        Args:
            models_dict (dict): Dictionary of trained models
            test_loader: Test data loader
            preprocessor: Data preprocessor for denormalization

        Returns:
            dict: Comparison results
        """
        print("Comparing models...")

        # Evaluate each model
        for model_name, model in models_dict.items():
            self.evaluate_single_model(model, test_loader, model_name, preprocessor)

        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            for metric_name, value in metrics.items():
                comparison_data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': value
                })

        comparison_df = pd.DataFrame(comparison_data)
        self.comparisons = comparison_df

        return comparison_df

    def get_best_model(self):
        """
        Determine the best performing model based on RMSE

        Returns:
            tuple: (best_model_name, best_rmse_value)
        """
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluation first.")

        best_model = None
        best_rmse = float('inf')

        for model_name, result in self.results.items():
            rmse = result['metrics']['RMSE']
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name

        return best_model, best_rmse

    def plot_trajectory_comparison(self, sample_index=0, feature_index=0):
        """
        Plot trajectory comparison for a sample

        Args:
            sample_index (int): Index of sample to plot
            feature_index (int): Index of feature to plot
        """
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluation first.")

        plt.figure(figsize=(12, 8))

        # Plot for each model
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        color_idx = 0

        for model_name, result in self.results.items():
            predictions = result['predictions']
            targets = result['targets']

            # Extract sample
            pred_sample = predictions[sample_index, :, feature_index]
            target_sample = targets[sample_index, :, feature_index]

            # Time steps
            pred_steps = np.arange(len(pred_sample))
            target_steps = np.arange(len(target_sample))

            # Plot
            plt.plot(target_steps, target_sample, '--',
                     color=colors[color_idx], label=f'{model_name} (Target)', linewidth=2)
            plt.plot(pred_steps, pred_sample, '-',
                     color=colors[color_idx], label=f'{model_name} (Prediction)', linewidth=2)

            color_idx = (color_idx + 1) % len(colors)

        plt.xlabel('Time Steps')
        plt.ylabel('Feature Value')
        plt.title('Trajectory Prediction Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()

    def plot_metrics_comparison(self):
        """
        Plot comparison of metrics across models

        Returns:
            matplotlib.figure.Figure: Comparison plot
        """
        if self.comparisons.empty:
            raise ValueError("No comparison data found. Run compare_models first.")

        # Pivot data for plotting
        pivot_df = self.comparisons.pivot(index='Metric', columns='Model', values='Value')

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_ylabel('Metric Value')
        ax.set_title('Model Performance Comparison')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_error_distribution(self, model_name):
        """
        Plot error distribution for a specific model

        Args:
            model_name (str): Name of model to analyze

        Returns:
            matplotlib.figure.Figure: Error distribution plot
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for model: {model_name}")

        result = self.results[model_name]
        predictions = result['predictions']
        targets = result['targets']

        # Calculate errors
        errors = predictions - targets
        errors_flat = errors.flatten()

        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Distribution - {model_name.upper()} Model')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def generate_report(self, save_path=None):
        """
        Generate comprehensive evaluation report

        Args:
            save_path (str): Path to save report (optional)

        Returns:
            str: Report content
        """
        if not self.results:
            raise ValueError("No evaluation results found. Run evaluation first.")

        report = []
        report.append("Q-PILOT MODEL EVALUATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated on: {pd.Timestamp.now()}")
        report.append("")

        # Model metrics
        report.append("MODEL PERFORMANCE METRICS:")
        report.append("-" * 30)
        for model_name, result in self.results.items():
            report.append(f"\n{model_name.upper()} MODEL:")
            metrics = result['metrics']
            for metric_name, value in metrics.items():
                report.append(f"  {metric_name}: {value:.6f}")

        # Best model
        best_model, best_rmse = self.get_best_model()
        report.append(f"\nBEST PERFORMING MODEL: {best_model.upper()}")
        report.append(f"BEST RMSE: {best_rmse:.6f}")

        # Comparison summary
        if not self.comparisons.empty:
            report.append("\nCOMPARISON SUMMARY:")
            report.append("-" * 20)
            summary = self.comparisons.groupby('Model')['Value'].mean()
            for model, avg_metric in summary.items():
                report.append(f"  {model}: {avg_metric:.6f}")

        report_content = "\n".join(report)

        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            print(f"Report saved to {save_path}")

        return report_content


class RealTimeComparator:
    """
    Real-time comparison engine for live model evaluation
    """
    def __init__(self, models_dict):
        """
        Initialize comparator

        Args:
            models_dict (dict): Dictionary of trained models
        """
        self.models = models_dict
        self.predictions = {}
        self.metrics_history = {}

    def run_comparison(self, input_sequence):
        """
        Run real-time comparison on input sequence

        Args:
            input_sequence (np.array): Input sequence data

        Returns:
            dict: Real-time predictions and metrics
        """
        # Ensure input is the right shape
        if len(input_sequence.shape) == 2:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        results = {}

        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                pred = None
                if model_name in ['linear', 'random_forest', 'lstm']:
                    # Try model-specific methods first for ensemble objects
                    method_name = f"predict_{model_name}"
                    if hasattr(model, method_name):
                        pred = getattr(model, method_name)(input_sequence)
                    elif hasattr(model, 'predict'):
                        pred = model.predict(input_sequence)
                
                # If still no prediction, try the fallback/PyTorch logic
                if pred is None:
                    # Quantum and other PyTorch models
                    batch_size = input_sequence.shape[0]
                    input_flat = input_sequence.reshape(batch_size, -1)
                    input_tensor = torch.FloatTensor(input_flat)

                    if hasattr(model, 'eval'):
                        model.eval()
                    
                    with torch.no_grad():
                        if callable(model):
                            pred_flat = model(input_tensor).numpy()
                            pred = pred_flat.reshape(batch_size, -1, input_sequence.shape[2])
                        elif hasattr(model, 'predict'):
                            pred = model.predict(input_sequence)

                if pred is not None:
                    results[model_name] = pred
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                results[model_name] = None

        self.predictions = results
        return results

    def rank_models(self, ground_truth=None):
        """
        Rank models based on performance

        Args:
            ground_truth (np.array): Ground truth values for ranking

        Returns:
            list: Ranked model names
        """
        if not self.predictions:
            raise ValueError("No predictions found. Run run_comparison first.")

        if ground_truth is None:
            # If no ground truth, rank by prediction confidence (simplified)
            rankings = list(self.predictions.keys())
            return rankings

        # Calculate metrics for each model
        model_scores = {}
        for model_name, predictions in self.predictions.items():
            if predictions is not None:
                try:
                    metrics = calculate_metrics(predictions, ground_truth)
                    # Use negative RMSE for ranking (higher is better)
                    model_scores[model_name] = -metrics['RMSE']
                except Exception as e:
                    print(f"Error calculating metrics for {model_name}: {e}")
                    model_scores[model_name] = float('-inf')
            else:
                model_scores[model_name] = float('-inf')

        # Sort by score
        ranked_models = sorted(model_scores.keys(), key=lambda x: model_scores[x], reverse=True)
        return ranked_models

    def get_winner(self, ground_truth=None):
        """
        Get the winning model

        Args:
            ground_truth (np.array): Ground truth values

        Returns:
            str: Name of winning model
        """
        rankings = self.rank_models(ground_truth)
        return rankings[0] if rankings else None


def load_and_evaluate_saved_models(model_paths, test_loader):
    """
    Load saved models and evaluate them

    Args:
        model_paths (dict): Paths to saved models
        test_loader: Test data loader

    Returns:
        dict: Evaluation results
    """
    # This function would load models from disk and evaluate them
    # Implementation would depend on how models are saved
    pass


def main():
    """
    Main evaluation function for demonstration
    """
    print("Model evaluation module ready for use.")
    print("Use ModelEvaluator class to evaluate trained models.")


if __name__ == "__main__":
    main()