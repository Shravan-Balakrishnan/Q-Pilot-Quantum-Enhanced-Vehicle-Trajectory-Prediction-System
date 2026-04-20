"""
Dataset handling for Q-Pilot system.
Supports both NGSIM dataset and synthetic data generation.
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os


class TrajectoryDataset(Dataset):
    """
    Custom Dataset for vehicle trajectory data
    """
    def __init__(self, data, seq_length=5, pred_length=3):
        """
        Initialize dataset

        Args:
            data (np.array): Input data of shape (samples, features)
            seq_length (int): Length of input sequence (T)
            pred_length (int): Length of prediction sequence (K)
        """
        self.seq_length = seq_length
        self.pred_length = pred_length

        # Create sequences
        self.X, self.y = self._create_sequences(data)

    def _create_sequences(self, data):
        """
        Create input-output sequence pairs

        Args:
            data (np.array): Input data

        Returns:
            tuple: (input_sequences, target_sequences)
        """
        X, y = [], []
        for i in range(len(data) - self.seq_length - self.pred_length + 1):
            X.append(data[i:(i + self.seq_length)])
            y.append(data[(i + self.seq_length):(i + self.seq_length + self.pred_length)])

        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].astype(np.float32), self.y[idx].astype(np.float32)


class NGSIMDataset:
    """
    Handler for NGSIM dataset
    """
    def __init__(self, data_path):
        """
        Initialize NGSIM dataset handler

        Args:
            data_path (str): Path to NGSIM dataset files
        """
        self.data_path = data_path
        self.data = None

    def load_data(self, file_name="ngsim.csv"):
        """
        Load NGSIM dataset

        Args:
            file_name (str): Name of the dataset file

        Returns:
            pd.DataFrame: Loaded dataset
        """
        file_path = os.path.join(self.data_path, file_name)
        if os.path.exists(file_path):
            self.data = pd.read_csv(file_path)
            return self.data
        else:
            print(f"Warning: {file_path} not found. Generating synthetic data instead.")
            self.data = self.generate_synthetic_data()
            return self.data

    def preprocess_data(self):
        """
        Preprocess NGSIM data to extract relevant features

        Returns:
            np.array: Processed data array of shape (samples, features)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Extract relevant columns
        # Assuming NGSIM has columns: Vehicle_ID, Frame_ID, Local_X, Local_Y, v_Vel, v_Acc, Lane_ID
        required_columns = ['Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Lane_ID']

        # Check if all required columns exist
        if not all(col in self.data.columns for col in required_columns):
            print("Warning: Some required columns not found. Using available columns.")
            available_columns = [col for col in required_columns if col in self.data.columns]
            required_columns = available_columns

        # Extract data
        processed_data = self.data[required_columns].values

        # Add synthetic steering angle if not present
        if 'Steering_Angle' not in self.data.columns:
            # Generate synthetic steering angle based on position changes
            steering_angles = self._calculate_steering_angles(processed_data)
            processed_data = np.column_stack([processed_data, steering_angles])

        return processed_data

    def _calculate_steering_angles(self, data):
        """
        Calculate synthetic steering angles from position data

        Args:
            data (np.array): Position and velocity data

        Returns:
            np.array: Calculated steering angles
        """
        # Simple approximation: steering angle proportional to heading change
        # In practice, this would be more complex
        steering_angles = np.zeros(len(data))
        if len(data) > 1:
            # Calculate heading changes
            dx = np.diff(data[:, 0], prepend=data[0, 0])  # x_position changes
            dy = np.diff(data[:, 1], prepend=data[0, 1])  # y_position changes

            # Calculate heading angles
            headings = np.arctan2(dy, dx)

            # Steering angle is change in heading
            steering_angles[1:] = np.diff(headings, prepend=headings[0])

        return steering_angles


def generate_synthetic_trajectory(n_points=1000, noise_level=0.1):
    """
    Generate synthetic vehicle trajectory data

    Args:
        n_points (int): Number of data points to generate
        noise_level (float): Level of noise to add to the data

    Returns:
        np.array: Synthetic trajectory data of shape (n_points, 6)
                  Columns: x_pos, y_pos, velocity, acceleration, steering_angle, lane_id
    """
    # Time vector
    t = np.linspace(0, 100, n_points)

    # Generate smooth trajectory (circular path with variations)
    radius = 50 + 10 * np.sin(0.1 * t)  # Variable radius
    angle = 0.1 * t + 0.02 * np.sin(0.05 * t)  # Angle with variations

    # Position
    x_pos = radius * np.cos(angle)
    y_pos = radius * np.sin(angle)

    # Velocity (derivative of position)
    dx = np.diff(x_pos, prepend=x_pos[0])
    dy = np.diff(y_pos, prepend=y_pos[0])
    velocity = np.sqrt(dx**2 + dy**2)

    # Acceleration (derivative of velocity)
    acceleration = np.diff(velocity, prepend=velocity[0])

    # Steering angle (derivative of heading)
    heading = np.arctan2(dy, dx)
    steering_angle = np.diff(heading, prepend=heading[0])

    # Lane ID (alternating lanes)
    lane_id = np.sin(0.05 * t) > 0  # Boolean mask for lane changes
    lane_id = lane_id.astype(int)  # Convert to integers

    # Combine into dataset
    data = np.column_stack([
        x_pos,
        y_pos,
        velocity,
        acceleration,
        steering_angle,
        lane_id
    ])

    # Add noise
    noise = np.random.normal(0, noise_level, data.shape)
    data_noisy = data + noise

    return data_noisy


def create_dataloader(data, seq_length=5, pred_length=3, batch_size=32, shuffle=True):
    """
    Create DataLoader for trajectory data

    Args:
        data (np.array): Input data
        seq_length (int): Length of input sequence
        pred_length (int): Length of prediction sequence
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the data

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = TrajectoryDataset(data, seq_length, pred_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    # Example usage
    # Generate synthetic data
    synthetic_data = generate_synthetic_trajectory(1000)
    print(f"Synthetic data shape: {synthetic_data.shape}")

    # Create dataloader
    dataloader = create_dataloader(synthetic_data)
    print(f"Number of batches: {len(dataloader)}")

    # Show sample batch
    for batch_x, batch_y in dataloader:
        print(f"Input batch shape: {batch_x.shape}")
        print(f"Target batch shape: {batch_y.shape}")
        break