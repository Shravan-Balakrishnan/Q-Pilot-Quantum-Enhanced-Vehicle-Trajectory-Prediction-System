"""
Data preprocessing module for Q-Pilot system.
Handles data cleaning, normalization, and sequence generation.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d


class DataPreprocessor:
    """
    Preprocessing pipeline for trajectory data
    """
    def __init__(self):
        self.scalers = {}
        self.feature_names = None

    def handle_missing_values(self, data, method='interpolate'):
        """
        Handle missing values in the dataset

        Args:
            data (np.array): Input data with potential missing values
            method (str): Method to handle missing values ('interpolate', 'forward_fill', 'backward_fill')

        Returns:
            np.array: Data with missing values handled
        """
        if method == 'interpolate':
            # Linear interpolation for missing values
            df = pd.DataFrame(data)
            df_interpolated = df.interpolate(method='linear', limit_direction='both')
            return df_interpolated.values
        elif method == 'forward_fill':
            df = pd.DataFrame(data)
            return df.fillna(method='ffill').fillna(method='bfill').values
        elif method == 'backward_fill':
            df = pd.DataFrame(data)
            return df.fillna(method='bfill').fillna(method='ffill').values
        else:
            # Return as is if method not recognized
            return data

    def normalize_features(self, data, method='minmax', feature_names=None):
        """
        Normalize features in the dataset

        Args:
            data (np.array): Input data
            method (str): Normalization method ('minmax', 'standard')
            feature_names (list): Names of features for scaler tracking

        Returns:
            np.array: Normalized data
        """
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(data.shape[1])]

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Fit and transform the data
        normalized_data = scaler.fit_transform(data)

        # Store scaler for each feature
        for i, name in enumerate(self.feature_names):
            self.scalers[name] = scaler

        return normalized_data

    def denormalize_features(self, data, feature_names=None):
        """
        Denormalize features using stored scalers

        Args:
            data (np.array): Normalized data
            feature_names (list): Names of features to denormalize

        Returns:
            np.array: Denormalized data
        """
        if not self.scalers:
            raise ValueError("No scalers found. Call normalize_features first.")

        feature_names = feature_names or self.feature_names
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]

        denormalized_data = np.copy(data)
        for i, name in enumerate(feature_names):
            if name in self.scalers:
                scaler = self.scalers[name]
                # Reshape for single feature inverse transform
                feature_data = data[:, i].reshape(-1, 1)
                denormalized_feature = scaler.inverse_transform(feature_data)
                denormalized_data[:, i] = denormalized_feature.flatten()

        return denormalized_data

    def create_sequences(self, data, seq_length, pred_length):
        """
        Create sequences for time series prediction

        Args:
            data (np.array): Input data of shape (samples, features)
            seq_length (int): Length of input sequence (T)
            pred_length (int): Length of prediction sequence (K)

        Returns:
            tuple: (input_sequences, target_sequences)
        """
        X, y = [], []
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[(i + seq_length):(i + seq_length + pred_length)])

        return np.array(X), np.array(y)

    def resample_trajectory(self, trajectory, target_points):
        """
        Resample trajectory to have consistent number of points

        Args:
            trajectory (np.array): Input trajectory data
            target_points (int): Target number of points

        Returns:
            np.array: Resampled trajectory
        """
        if len(trajectory) == target_points:
            return trajectory

        # Create interpolation functions for each feature
        x_points = np.linspace(0, 1, len(trajectory))
        x_new = np.linspace(0, 1, target_points)

        resampled = np.zeros((target_points, trajectory.shape[1]))

        for i in range(trajectory.shape[1]):
            f = interp1d(x_points, trajectory[:, i], kind='linear')
            resampled[:, i] = f(x_new)

        return resampled


def prepare_data_pipeline(data, seq_length=5, pred_length=3, normalize_method='minmax'):
    """
    Complete data preparation pipeline

    Args:
        data (np.array): Raw input data
        seq_length (int): Length of input sequence
        pred_length (int): Length of prediction sequence
        normalize_method (str): Normalization method

    Returns:
        tuple: (processed_data, scaler_info)
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Handle missing values
    data_clean = preprocessor.handle_missing_values(data, method='interpolate')

    # Normalize features
    data_normalized = preprocessor.normalize_features(
        data_clean,
        method=normalize_method,
        feature_names=['x_position', 'y_position', 'velocity', 'acceleration', 'steering_angle', 'lane_id']
    )

    # Create sequences
    X, y = preprocessor.create_sequences(data_normalized, seq_length, pred_length)

    # Return processed data and preprocessor for later use
    return (X, y), preprocessor


if __name__ == "__main__":
    # Example usage with synthetic data
    from dataset import generate_synthetic_trajectory

    # Generate sample data
    raw_data = generate_synthetic_trajectory(1000)
    print(f"Raw data shape: {raw_data.shape}")

    # Process data
    (X, y), preprocessor = prepare_data_pipeline(raw_data)
    print(f"Processed input shape: {X.shape}")
    print(f"Processed target shape: {y.shape}")

    # Denormalize sample
    sample_normalized = X[0]
    sample_denormalized = preprocessor.denormalize_features(sample_normalized)
    print(f"Sample denormalized shape: {sample_denormalized.shape}")