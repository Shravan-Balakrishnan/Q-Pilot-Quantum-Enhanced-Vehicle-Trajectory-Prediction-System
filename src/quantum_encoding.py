"""
Quantum encoding methods for Q-Pilot system.
Implements various quantum data encoding techniques.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.kernels import QuantumKernel
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class AngleEncoding:
    """
    Angle encoding for quantum data representation
    Encodes classical data into quantum states using rotation gates
    """
    def __init__(self, num_qubits, encoding_type='ry'):
        """
        Initialize angle encoder

        Args:
            num_qubits (int): Number of qubits to use
            encoding_type (str): Type of rotation gates ('rx', 'ry', 'rz')
        """
        self.num_qubits = num_qubits
        self.encoding_type = encoding_type.lower()

    def encode(self, data):
        """
        Encode classical data into quantum circuit

        Args:
            data (np.array): Input data to encode

        Returns:
            QuantumCircuit: Quantum circuit with encoded data
        """
        # Normalize data to [0, π] for rotation angles
        normalized_data = self._normalize_data(data)

        # Create quantum circuit
        qc = QuantumCircuit(self.num_qubits)

        # Apply rotation gates based on encoding type
        if self.encoding_type == 'rx':
            for i, val in enumerate(normalized_data[:self.num_qubits]):
                qc.rx(val, i)
        elif self.encoding_type == 'ry':
            for i, val in enumerate(normalized_data[:self.num_qubits]):
                qc.ry(val, i)
        elif self.encoding_type == 'rz':
            for i, val in enumerate(normalized_data[:self.num_qubits]):
                qc.rz(val, i)
        else:
            raise ValueError(f"Unsupported encoding type: {self.encoding_type}")

        return qc

    def _normalize_data(self, data):
        """
        Normalize data to [0, π] range for rotation angles

        Args:
            data (np.array): Input data

        Returns:
            np.array: Normalized data
        """
        # Handle case where data has multiple features
        if len(data.shape) > 1:
            data = data.flatten()

        # Normalize to [0, π]
        data_min, data_max = np.min(data), np.max(data)
        if data_max - data_min != 0:
            normalized = (data - data_min) / (data_max - data_min) * np.pi
        else:
            normalized = np.zeros_like(data)

        return normalized


class AmplitudeEncoding:
    """
    Amplitude encoding for quantum data representation
    Encodes classical data into amplitudes of quantum states
    """
    def __init__(self, num_qubits):
        """
        Initialize amplitude encoder

        Args:
            num_qubits (int): Number of qubits to use
        """
        self.num_qubits = num_qubits
        self.max_features = 2**num_qubits

    def encode(self, data):
        """
        Encode classical data into quantum state amplitudes

        Args:
            data (np.array): Input data to encode (should have at most 2^num_qubits elements)

        Returns:
            QuantumCircuit: Quantum circuit with encoded data
        """
        # Pad or truncate data to match state dimension
        padded_data = self._pad_data(data)

        # Normalize to create valid quantum state
        normalized_data = padded_data / np.linalg.norm(padded_data)

        # Create quantum circuit
        qc = QuantumCircuit(self.num_qubits)

        # Initialize quantum state with amplitudes
        qc.initialize(normalized_data, range(self.num_qubits))

        return qc

    def _pad_data(self, data):
        """
        Pad or truncate data to match required dimension

        Args:
            data (np.array): Input data

        Returns:
            np.array: Padded/truncated data
        """
        # Handle multidimensional data
        if len(data.shape) > 1:
            data = data.flatten()

        # Pad or truncate to match 2^num_qubits
        if len(data) < self.max_features:
            padded = np.pad(data, (0, self.max_features - len(data)))
        else:
            padded = data[:self.max_features]

        return padded


class QuantumFeatureMap:
    """
    Quantum feature map for data encoding
    Uses Qiskit's built-in feature maps
    """
    def __init__(self, num_qubits, feature_map_type='zz', reps=2):
        """
        Initialize quantum feature map

        Args:
            num_qubits (int): Number of qubits
            feature_map_type (str): Type of feature map ('zz', 'real_amplitudes')
            reps (int): Number of repetitions
        """
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map_type
        self.reps = reps

        # Create feature map
        if feature_map_type == 'zz':
            self.feature_map = ZZFeatureMap(
                feature_dimension=num_qubits,
                reps=reps,
                entanglement='linear'
            )
        elif feature_map_type == 'real_amplitudes':
            self.feature_map = RealAmplitudes(
                num_qubits=num_qubits,
                reps=reps,
                entanglement='linear'
            )
        else:
            raise ValueError(f"Unsupported feature map type: {feature_map_type}")

    def encode(self, data):
        """
        Encode data using quantum feature map

        Args:
            data (np.array): Input data

        Returns:
            QuantumCircuit: Quantum circuit with encoded data
        """
        # Normalize data to [0, 1]
        normalized_data = self._normalize_data(data)

        # Bind parameters to feature map
        # For ZZFeatureMap, we need to match dimensions
        if len(normalized_data) > self.num_qubits:
            # Truncate if more features than qubits
            param_data = normalized_data[:self.num_qubits]
        else:
            # Pad if fewer features than qubits
            param_data = np.pad(normalized_data, (0, self.num_qubits - len(normalized_data)))

        # Create circuit with bound parameters
        qc = self.feature_map.assign_parameters(param_data)
        return qc

    def _normalize_data(self, data):
        """
        Normalize data to [0, 1] range

        Args:
            data (np.array): Input data

        Returns:
            np.array: Normalized data
        """
        # Handle multidimensional data
        if len(data.shape) > 1:
            data = data.flatten()

        # Normalize to [0, 1]
        data_min, data_max = np.min(data), np.max(data)
        if data_max - data_min != 0:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)

        return normalized


class EncoderFactory:
    """
    Factory for creating different types of quantum encoders
    """
    @staticmethod
    def create_encoder(encoder_type, num_qubits, **kwargs):
        """
        Create quantum encoder of specified type

        Args:
            encoder_type (str): Type of encoder ('angle', 'amplitude', 'feature_map')
            num_qubits (int): Number of qubits
            **kwargs: Additional arguments for specific encoders

        Returns:
            QuantumEncoder: Instance of requested encoder
        """
        if encoder_type == 'angle':
            encoding_type = kwargs.get('encoding_type', 'ry')
            return AngleEncoding(num_qubits, encoding_type)
        elif encoder_type == 'amplitude':
            return AmplitudeEncoding(num_qubits)
        elif encoder_type == 'feature_map':
            feature_map_type = kwargs.get('feature_map_type', 'zz')
            reps = kwargs.get('reps', 2)
            return QuantumFeatureMap(num_qubits, feature_map_type, reps)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


def compare_encodings(data_sample, num_qubits=4):
    """
    Compare different quantum encoding methods

    Args:
        data_sample (np.array): Sample data to encode
        num_qubits (int): Number of qubits to use

    Returns:
        dict: Dictionary with circuits for each encoding method
    """
    encodings = {}

    # Angle encoding (RY)
    angle_encoder = AngleEncoding(num_qubits, 'ry')
    encodings['angle_ry'] = angle_encoder.encode(data_sample)

    # Angle encoding (RX)
    angle_encoder_rx = AngleEncoding(num_qubits, 'rx')
    encodings['angle_rx'] = angle_encoder_rx.encode(data_sample)

    # Angle encoding (RZ)
    angle_encoder_rz = AngleEncoding(num_qubits, 'rz')
    encodings['angle_rz'] = angle_encoder_rz.encode(data_sample)

    # Feature map encoding
    feature_encoder = QuantumFeatureMap(num_qubits, 'zz')
    encodings['feature_map'] = feature_encoder.encode(data_sample)

    return encodings


if __name__ == "__main__":
    # Example usage
    # Generate sample data
    sample_data = np.random.rand(4)  # 4 features for 4 qubits
    print(f"Sample data: {sample_data}")

    # Compare encodings
    encodings = compare_encodings(sample_data)

    for name, circuit in encodings.items():
        print(f"\n{name.upper()} ENCODING:")
        print(circuit.draw(fold=-1))