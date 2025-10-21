"""
Variable-depth MLP models for MNIST.

Studies how the number of hidden layers affects generalization
within the same architecture family (fully-connected MLPs).

Architecture uses fixed 256 neurons for all hidden layers.
"""
import torch
import torch.nn as nn


class MLP_Variable(nn.Module):
    """
    Variable-depth MLP for MNIST classification with fixed 256 neurons per layer.

    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden layers: All layers have 256 neurons
    - Optional BatchNorm after each hidden layer
    - Output: 10 classes

    Args:
        num_classes (int): Number of output classes (default: 10)
        n_layers (int): Number of hidden layers (1-50)
        initial_hidden_dim (int): Hidden dimension for all layers (default: 256)
        with_bn (bool): Whether to use BatchNorm (default: True)
    """
    def __init__(self, num_classes=10, n_layers=3, initial_hidden_dim=256, with_bn=True):
        super(MLP_Variable, self).__init__()

        if n_layers < 1 or n_layers > 50:
            raise ValueError(f"n_layers must be between 1 and 50, got {n_layers}")

        self.n_layers = n_layers
        self.initial_hidden_dim = initial_hidden_dim
        self.input_dim = 784  # 28x28 MNIST
        self.with_bn = with_bn

        # Build the network with fixed hidden dimensions
        layers = []

        # All hidden layers have the same dimension
        hidden_dim = initial_hidden_dim

        # First hidden layer: 784 -> hidden_dim -> (BN) -> ReLU
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        if with_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Additional hidden layers with fixed dimensions
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if with_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: hidden_dim -> num_classes (no BN or ReLU)
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming (He) initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Flatten input: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_first_hidden_layer(self):
        """Get the module for first hidden layer output (for MI evaluation).

        Returns the ReLU activation after the first hidden layer.
        This is used to hook and mask neurons in the first hidden layer.

        Raises:
            ValueError: If n_layers < 1 (no hidden layers exist).
        """
        if self.n_layers < 1:
            raise ValueError(f"Masking requires n_layers >= 1. Got n_layers={self.n_layers}")

        # For MLP_Variable, the network is a Sequential module
        # First hidden layer structure: Linear(784, h) -> BN (optional) -> ReLU
        # We want to return the ReLU module after the first hidden layer
        if self.with_bn:
            # Structure: Linear -> BN -> ReLU -> [more layers...]
            # Return the ReLU at index 2 (0=Linear, 1=BN, 2=ReLU)
            return self.network[2]
        else:
            # Structure: Linear -> ReLU -> [more layers...]
            # Return the ReLU at index 1 (0=Linear, 1=ReLU)
            return self.network[1]


def MLP1_Variable(num_classes=10, n_layers=1, **kwargs):
    """MLP with 1 hidden layer (256 neurons)."""
    if n_layers != 1:
        raise ValueError(f"MLP1_Variable expects n_layers=1, got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=1, **kwargs)


def MLP2_Variable(num_classes=10, n_layers=2, **kwargs):
    """MLP with up to 2 hidden layers (256 neurons each)."""
    if n_layers < 1 or n_layers > 2:
        raise ValueError(f"MLP2_Variable expects n_layers in [1,2], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP3_Variable(num_classes=10, n_layers=3, **kwargs):
    """MLP with up to 3 hidden layers (256 neurons each)."""
    if n_layers < 1 or n_layers > 3:
        raise ValueError(f"MLP3_Variable expects n_layers in [1,3], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP8_Variable(num_classes=10, n_layers=8, **kwargs):
    """MLP with up to 8 hidden layers (256 neurons each)."""
    if n_layers < 1 or n_layers > 8:
        raise ValueError(f"MLP8_Variable expects n_layers in [1,8], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP11_Variable(num_classes=10, n_layers=11, **kwargs):
    """MLP with up to 11 hidden layers (256 neurons each)."""
    if n_layers < 1 or n_layers > 11:
        raise ValueError(f"MLP11_Variable expects n_layers in [1,11], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP15_Variable(num_classes=10, n_layers=15, **kwargs):
    """MLP with up to 15 hidden layers (256 neurons each)."""
    if n_layers < 1 or n_layers > 15:
        raise ValueError(f"MLP15_Variable expects n_layers in [1,15], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


if __name__ == "__main__":
    # Test the models
    print("Testing MLP Variable with fixed 256 neurons:")
    print("=" * 80)
    for n in range(1, 16):
        model = MLP_Variable(n_layers=n)
        x = torch.randn(32, 1, 28, 28)
        y = model(x)
        hidden_str = f"{model.hidden_dim} × {n}"
        print(f"MLP with {n:2d} hidden layers: {hidden_str:40s} | params: {model.count_parameters():,}")
