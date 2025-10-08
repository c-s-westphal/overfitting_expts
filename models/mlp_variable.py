"""
Variable-depth MLP models for MNIST.

Studies how the number of hidden layers affects generalization
within the same architecture family (fully-connected MLPs).
"""
import torch
import torch.nn as nn


class MLP_Variable(nn.Module):
    """
    Variable-depth MLP for MNIST classification.

    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden layers: n_layers x 256 with ReLU
    - Output: 10 classes

    Args:
        num_classes (int): Number of output classes (default: 10)
        n_layers (int): Number of hidden layers (1-5)
        hidden_dim (int): Width of hidden layers (default: 256)
    """
    def __init__(self, num_classes=10, n_layers=3, hidden_dim=256):
        super(MLP_Variable, self).__init__()

        if n_layers < 1 or n_layers > 5:
            raise ValueError(f"n_layers must be between 1 and 5, got {n_layers}")

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = 784  # 28x28 MNIST

        # Build the network
        layers = []

        # First hidden layer: 784 -> hidden_dim
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Additional hidden layers: hidden_dim -> hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: hidden_dim -> num_classes
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming (He) initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Flatten input: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        return self.network(x)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def MLP1_Variable(num_classes=10, n_layers=1, **kwargs):
    """MLP with 1 hidden layer (variable depth 1-1)."""
    if n_layers != 1:
        raise ValueError(f"MLP1_Variable expects n_layers=1, got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=1, **kwargs)


def MLP2_Variable(num_classes=10, n_layers=2, **kwargs):
    """MLP with up to 2 hidden layers (variable depth 1-2)."""
    if n_layers < 1 or n_layers > 2:
        raise ValueError(f"MLP2_Variable expects n_layers in [1,2], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP3_Variable(num_classes=10, n_layers=3, **kwargs):
    """MLP with up to 3 hidden layers (variable depth 1-3)."""
    if n_layers < 1 or n_layers > 3:
        raise ValueError(f"MLP3_Variable expects n_layers in [1,3], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP4_Variable(num_classes=10, n_layers=4, **kwargs):
    """MLP with up to 4 hidden layers (variable depth 1-4)."""
    if n_layers < 1 or n_layers > 4:
        raise ValueError(f"MLP4_Variable expects n_layers in [1,4], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP5_Variable(num_classes=10, n_layers=5, **kwargs):
    """MLP with up to 5 hidden layers (variable depth 1-5)."""
    if n_layers < 1 or n_layers > 5:
        raise ValueError(f"MLP5_Variable expects n_layers in [1,5], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


if __name__ == "__main__":
    # Test the models
    for n in range(1, 6):
        model = MLP_Variable(n_layers=n)
        x = torch.randn(32, 1, 28, 28)
        y = model(x)
        print(f"MLP with {n} hidden layers: input {x.shape} -> output {y.shape}, params: {model.count_parameters():,}")
