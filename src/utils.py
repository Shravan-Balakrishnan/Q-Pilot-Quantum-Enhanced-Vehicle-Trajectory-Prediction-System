"""
Utility functions for the Q-Pilot system.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score


def calculate_ade(predictions, targets):
    """
    Calculate Average Displacement Error (ADE)

    Args:
        predictions (np.array): Predicted trajectories of shape (batch, timesteps, features)
        targets (np.array): Ground truth trajectories of shape (batch, timesteps, features)

    Returns:
        float: Average displacement error
    """
    # Calculate Euclidean distance for each time step
    displacement = np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))
    # Average across time steps and batch
    ade = np.mean(displacement)
    return ade


def calculate_fde(predictions, targets):
    """
    Calculate Final Displacement Error (FDE)

    Args:
        predictions (np.array): Predicted trajectories of shape (batch, timesteps, features)
        targets (np.array): Ground truth trajectories of shape (batch, timesteps, features)

    Returns:
        float: Final displacement error
    """
    # Calculate Euclidean distance for final time step only
    final_pred = predictions[:, -1, :]
    final_target = targets[:, -1, :]
    fde = np.sqrt(np.sum((final_pred - final_target) ** 2, axis=-1))
    return np.mean(fde)


def calculate_rmse(predictions, targets):
    """
    Calculate Root Mean Square Error (RMSE)

    Args:
        predictions (np.array): Predicted values
        targets (np.array): Ground truth values

    Returns:
        float: Root mean square error
    """
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    return rmse


def calculate_metrics(predictions, targets):
    """
    Calculate all evaluation metrics

    Args:
        predictions (np.array): Predicted values
        targets (np.array): Ground truth values

    Returns:
        dict: Dictionary containing all metrics
    """
    # Flatten arrays for metric calculations
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    target_flat = targets.reshape(-1, targets.shape[-1])

    # Calculate metrics
    mse = mean_squared_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(target_flat, pred_flat)
    ade = calculate_ade(predictions, targets)
    fde = calculate_fde(predictions, targets)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'ADE': ade,
        'FDE': fde
    }


def normalize_data(data, feature_range=(0, 1)):
    """
    Normalize data to specified feature range

    Args:
        data (np.array): Input data
        feature_range (tuple): Min and max values for normalization

    Returns:
        np.array: Normalized data
        dict: Min and max values for denormalization
    """
    min_val, max_val = feature_range
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    # Avoid division by zero
    data_range = data_max - data_min
    data_range[data_range == 0] = 1

    normalized = (data - data_min) / data_range * (max_val - min_val) + min_val

    return normalized, {'min': data_min, 'max': data_max}


def denormalize_data(normalized_data, norm_params, feature_range=(0, 1)):
    """
    Denormalize data using normalization parameters

    Args:
        normalized_data (np.array): Normalized data
        norm_params (dict): Dictionary with 'min' and 'max' keys
        feature_range (tuple): Original feature range used in normalization

    Returns:
        np.array: Denormalized data
    """
    min_val, max_val = feature_range
    data_min = norm_params['min']
    data_max = norm_params['max']

    data_range = data_max - data_min
    data_range[data_range == 0] = 1

    denormalized = (normalized_data - min_val) / (max_val - min_val) * data_range + data_min

    return denormalized


def create_sequences(data, seq_length, pred_length):
    """
    Create sequences for time series prediction

    Args:
        data (np.array): Input data of shape (samples, features)
        seq_length (int): Length of input sequence
        pred_length (int): Length of prediction sequence

    Returns:
        tuple: (input_sequences, target_sequences)
    """
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + pred_length)])

    return np.array(X), np.array(y)


def save_model(model, path):
    """
    Save PyTorch model

    Args:
        model (torch.nn.Module): Model to save
        path (str): Path to save model
    """
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """
    Load PyTorch model

    Args:
        model (torch.nn.Module): Model class to load into
        path (str): Path to saved model

    Returns:
        torch.nn.Module: Loaded model
    """
    model.load_state_dict(torch.load(path))
    return model


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    """
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0