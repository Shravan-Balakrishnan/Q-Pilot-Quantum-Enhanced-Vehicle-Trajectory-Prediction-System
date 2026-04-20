"""
Classical machine learning models for trajectory prediction.
Includes Linear Regression and LSTM models.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class LinearTrajectoryPredictor:
    """
    Linear regression model for trajectory prediction
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize linear predictor

        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Dimension of output features
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.models = [LinearRegression() for _ in range(output_dim)]

    def fit(self, X, y):
        """
        Train the model

        Args:
            X (np.array): Input data of shape (samples, seq_length, features)
            y (np.array): Target data of shape (samples, pred_length, features)
        """
        # Flatten the sequence dimension for linear regression
        X_flat = X.reshape(X.shape[0], -1)
        
        # Flatten the target time and feature dimensions
        # y_flat will be (samples, pred_length * features)
        y_flat = y.reshape(y.shape[0], -1)

        # Train a single multi-output model or separate models
        # For simplicity and to match the current structure, we'll use one model per output value
        # but with consistent sample counts
        for i in range(self.output_dim):
            self.models[i].fit(X_flat, y_flat[:, i])

    def predict(self, X):
        """
        Make predictions

        Args:
            X (np.array): Input data of shape (samples, seq_length, features)

        Returns:
            np.array: Predictions of shape (samples, pred_length, features)
        """
        X_flat = X.reshape(X.shape[0], -1)
        predictions = []

        # Predict for each output dimension
        for i in range(self.output_dim):
            pred = self.models[i].predict(X_flat)
            predictions.append(pred)

        # Stack and reshape to (samples, pred_length, features)
        all_preds = np.column_stack(predictions)
        return all_preds.reshape(X.shape[0], -1, X.shape[2])


class LSTMTrajectoryPredictor(nn.Module):
    """
    LSTM model for trajectory prediction
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        """
        Initialize LSTM predictor

        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden state
            output_dim (int): Dimension of output features
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
        """
        super(LSTMTrajectoryPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Activation for output (depending on data characteristics)
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, features)

        Returns:
            torch.Tensor: Output tensor of shape (batch, pred_length, output_dim)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output for prediction
        # Or use all outputs depending on requirements
        output = self.fc(lstm_out)
        output = self.activation(output)

        return output

    def predict(self, x):
        """
        Make predictions (for compatibility with sklearn-like interface)

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            np.array: Predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return output.cpu().numpy()


class MLPTrajectoryPredictor(nn.Module):
    """
    Multi-Layer Perceptron for trajectory prediction
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        """
        Initialize MLP predictor

        Args:
            input_dim (int): Dimension of flattened input
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Dimension of output
            dropout (float): Dropout rate
        """
        super(MLPTrajectoryPredictor, self).__init__()

        # Calculate flattened input dimension
        # Assuming input is (seq_length, features)
        self.input_dim = input_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, features)

        Returns:
            torch.Tensor: Output tensor
        """
        # Flatten input
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Forward through network
        output = self.network(x_flat)
        return output


class ClassicalModelEnsemble:
    """
    Ensemble of classical models for comparison
    """
    def __init__(self, input_shape, output_shape):
        """
        Initialize ensemble

        Args:
            input_shape (tuple): Shape of input data (seq_length, features)
            output_shape (tuple): Shape of output data (pred_length, features)
        """
        self.seq_length, self.input_features = input_shape
        self.pred_length, self.output_features = output_shape

        # Initialize models
        self.linear_model = LinearTrajectoryPredictor(
            input_dim=self.seq_length * self.input_features,
            output_dim=self.pred_length * self.output_features
        )

        self.lstm_model = LSTMTrajectoryPredictor(
            input_dim=self.input_features,
            hidden_dim=64,
            output_dim=self.output_features,
            num_layers=2
        )

        # Optional: Random Forest model
        self.rf_models = [RandomForestRegressor(n_estimators=100)
                         for _ in range(self.output_features)]

    def fit_linear(self, X, y):
        """
        Fit linear model

        Args:
            X (np.array): Input data
            y (np.array): Target data
        """
        self.linear_model.fit(X, y)

    def fit_lstm(self, train_loader, num_epochs=50, learning_rate=0.001):
        """
        Fit LSTM model

        Args:
            train_loader (DataLoader): Training data loader
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=learning_rate)

        self.lstm_model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                # Convert to tensors
                batch_x = torch.FloatTensor(batch_x)
                batch_y = torch.FloatTensor(batch_y)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'LSTM Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

    def fit_random_forest(self, X, y):
        """
        Fit Random Forest models

        Args:
            X (np.array): Input data
            y (np.array): Target data
        """
        # Flatten the sequence dimension
        X_flat = X.reshape(X.shape[0], -1)
        
        # Flatten target (samples, pred_length * features)
        y_flat = y.reshape(y.shape[0], -1)

        # Train separate model for each output dimension (pred_length * features)
        # Note: rf_models list must be long enough
        if len(self.rf_models) < y_flat.shape[1]:
            self.rf_models = [RandomForestRegressor(n_estimators=10) for _ in range(y_flat.shape[1])]

        for i in range(y_flat.shape[1]):
            self.rf_models[i].fit(X_flat, y_flat[:, i])

    def predict_linear(self, X):
        """
        Predict with linear model

        Args:
            X (np.array): Input data

        Returns:
            np.array: Predictions
        """
        return self.linear_model.predict(X)

    def predict_lstm(self, X):
        """
        Predict with LSTM model

        Args:
            X (np.array or torch.Tensor): Input data

        Returns:
            np.array: Predictions
        """
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)

        self.lstm_model.eval()
        with torch.no_grad():
            predictions = self.lstm_model(X)
            return predictions.numpy()

    def predict_random_forest(self, X):
        """
        Predict with Random Forest models

        Args:
            X (np.array): Input data

        Returns:
            np.array: Predictions
        """
        X_flat = X.reshape(X.shape[0], -1)
        predictions = []

        # Predict for each output dimension
        for i in range(len(self.rf_models)):
            pred = self.rf_models[i].predict(X_flat)
            predictions.append(pred)

        # Stack and reshape to (samples, pred_length, features)
        all_preds = np.column_stack(predictions)
        return all_preds.reshape(X.shape[0], -1, self.output_features)


if __name__ == "__main__":
    # Example usage
    from dataset import generate_synthetic_trajectory, create_dataloader

    # Generate sample data
    raw_data = generate_synthetic_trajectory(1000)

    # Create sequences
    seq_length = 5
    pred_length = 3
    X_data = []
    y_data = []

    for i in range(len(raw_data) - seq_length - pred_length + 1):
        X_data.append(raw_data[i:(i + seq_length)])
        y_data.append(raw_data[(i + seq_length):(i + seq_length + pred_length)])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    print(f"Input shape: {X_data.shape}")
    print(f"Target shape: {y_data.shape}")

    # Initialize ensemble
    ensemble = ClassicalModelEnsemble(
        input_shape=(seq_length, raw_data.shape[1]),
        output_shape=(pred_length, raw_data.shape[1])
    )

    # Fit linear model
    print("Training linear model...")
    ensemble.fit_linear(X_data, y_data)

    # Make prediction
    linear_pred = ensemble.predict_linear(X_data[:5])
    print(f"Linear prediction shape: {linear_pred.shape}")