"""
Feature engineering module for Q-Pilot system.
Extracts and creates additional features from raw trajectory data.
"""
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import PolynomialFeatures


class FeatureEngineer:
    """
    Feature engineering pipeline for trajectory data
    """
    def __init__(self):
        self.feature_names = [
            'x_position', 'y_position', 'velocity', 'acceleration',
            'steering_angle', 'lane_id'
        ]

    def extract_position_features(self, data):
        """
        Extract position-based features

        Args:
            data (np.array): Trajectory data with columns [x, y, ...]

        Returns:
            np.array: Additional position features
        """
        x_pos = data[:, 0]
        y_pos = data[:, 1]

        # Calculate distances between consecutive points
        distances = np.zeros(len(x_pos))
        for i in range(1, len(x_pos)):
            distances[i] = euclidean([x_pos[i-1], y_pos[i-1]], [x_pos[i], y_pos[i]])

        # Calculate cumulative distance
        cumulative_distance = np.cumsum(distances)

        # Calculate direction (angle)
        directions = np.zeros(len(x_pos))
        for i in range(1, len(x_pos)):
            dx = x_pos[i] - x_pos[i-1]
            dy = y_pos[i] - y_pos[i-1]
            directions[i] = np.arctan2(dy, dx)

        return np.column_stack([distances, cumulative_distance, directions])

    def extract_velocity_features(self, data):
        """
        Extract velocity-based features

        Args:
            data (np.array): Trajectory data with velocity column

        Returns:
            np.array: Additional velocity features
        """
        velocity = data[:, 2]  # Assuming velocity is 3rd column

        # Calculate jerk (derivative of acceleration)
        jerk = np.diff(velocity, prepend=velocity[0])

        # Calculate speed change rate
        speed_change_rate = np.abs(jerk)

        return np.column_stack([jerk, speed_change_rate])

    def extract_lane_features(self, data):
        """
        Extract lane-based features

        Args:
            data (np.array): Trajectory data with lane_id column

        Returns:
            np.array: Additional lane features
        """
        lane_id = data[:, 5]  # Assuming lane_id is 6th column

        # Detect lane changes
        lane_changes = np.zeros(len(lane_id))
        for i in range(1, len(lane_id)):
            if lane_id[i] != lane_id[i-1]:
                lane_changes[i] = 1

        # Count of lane changes in window
        lane_change_count = np.convolve(lane_changes, np.ones(5), mode='same')

        return np.column_stack([lane_changes, lane_change_count])

    def create_polynomial_features(self, data, degree=2):
        """
        Create polynomial features

        Args:
            data (np.array): Input data
            degree (int): Degree of polynomial features

        Returns:
            np.array: Polynomial features
        """
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(data)
        return poly_features

    def engineer_features(self, data):
        """
        Complete feature engineering pipeline

        Args:
            data (np.array): Raw trajectory data

        Returns:
            np.array: Engineered features
        """
        # Extract different types of features
        position_features = self.extract_position_features(data)
        velocity_features = self.extract_velocity_features(data)
        lane_features = self.extract_lane_features(data)

        # Combine all features
        engineered_data = np.column_stack([
            data,
            position_features,
            velocity_features,
            lane_features
        ])

        return engineered_data

    def get_feature_names(self):
        """
        Get names of all engineered features

        Returns:
            list: Feature names
        """
        extended_names = self.feature_names.copy()
        extended_names.extend([
            'distance', 'cumulative_distance', 'direction',
            'jerk', 'speed_change_rate',
            'lane_change', 'lane_change_count'
        ])
        return extended_names


def add_noise_features(data, noise_level=0.01):
    """
    Add noise features for robustness testing

    Args:
        data (np.array): Input data
        noise_level (float): Level of noise to add

    Returns:
        np.array: Data with noise features
    """
    noise = np.random.normal(0, noise_level, data.shape)
    noisy_data = data + noise
    return noisy_data


def create_interaction_features(data):
    """
    Create interaction features between variables

    Args:
        data (np.array): Input data

    Returns:
        np.array: Data with interaction features
    """
    # Create some meaningful interactions
    interactions = []

    if data.shape[1] >= 6:
        # Velocity * acceleration
        vel_acc = data[:, 2] * data[:, 3]
        interactions.append(vel_acc)

        # Steering angle * velocity
        steer_vel = data[:, 4] * data[:, 2]
        interactions.append(steer_vel)

        # Position combinations
        pos_product = data[:, 0] * data[:, 1]  # x*y
        interactions.append(pos_product)

    if interactions:
        interaction_matrix = np.column_stack(interactions)
        return np.column_stack([data, interaction_matrix])
    else:
        return data


def enhance_features(data):
    """
    Apply comprehensive feature enhancement

    Args:
        data (np.array): Input data

    Returns:
        np.array: Enhanced feature set
    """
    # Initialize feature engineer
    engineer = FeatureEngineer()

    # Engineer features
    enhanced_data = engineer.engineer_features(data)

    # Add interaction features
    enhanced_data = create_interaction_features(enhanced_data)

    # Add noise for robustness (small amount)
    enhanced_data = add_noise_features(enhanced_data, noise_level=0.001)

    return enhanced_data


if __name__ == "__main__":
    # Example usage
    from dataset import generate_synthetic_trajectory

    # Generate sample data
    raw_data = generate_synthetic_trajectory(100)
    print(f"Raw data shape: {raw_data.shape}")

    # Enhance features
    enhanced_data = enhance_features(raw_data)
    print(f"Enhanced data shape: {enhanced_data.shape}")

    # Show feature engineer
    engineer = FeatureEngineer()
    feature_names = engineer.get_feature_names()
    print(f"Base features: {feature_names}")