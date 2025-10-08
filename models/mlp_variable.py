"""
Variable-depth MLP models for MNIST.

Studies how the number of hidden layers affects generalization
within the same architecture family (fully-connected MLPs).

Architecture uses a halving pattern: 1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
"""
import torch
import torch.nn as nn


class MLP_Variable(nn.Module):
    """
    Variable-depth MLP for MNIST classification with halving architecture.

    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden layers: Starting at 1024, halving each layer (1024 → 512 → 256 → ...)
    - Output: 10 classes

    Args:
        num_classes (int): Number of output classes (default: 10)
        n_layers (int): Number of hidden layers (1-11)
        initial_hidden_dim (int): Starting hidden dimension (default: 1024)
    """
    def __init__(self, num_classes=10, n_layers=3, initial_hidden_dim=1024):
        super(MLP_Variable, self).__init__()

        if n_layers < 1 or n_layers > 11:
            raise ValueError(f"n_layers must be between 1 and 11, got {n_layers}")

        self.n_layers = n_layers
        self.initial_hidden_dim = initial_hidden_dim
        self.input_dim = 784  # 28x28 MNIST

        # Build the network with halving hidden dimensions
        layers = []

        # Calculate hidden layer dimensions (halving each time)
        hidden_dims = [initial_hidden_dim // (2 ** i) for i in range(n_layers)]

        # First hidden layer: 784 -> hidden_dims[0]
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(nn.ReLU(inplace=True))

        # Additional hidden layers with halving dimensions
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        # Output layer: last_hidden_dim -> num_classes
        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.network = nn.Sequential(*layers)
        self.hidden_dims = hidden_dims

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
    """MLP with 1 hidden layer (1024)."""
    if n_layers != 1:
        raise ValueError(f"MLP1_Variable expects n_layers=1, got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=1, **kwargs)


def MLP2_Variable(num_classes=10, n_layers=2, **kwargs):
    """MLP with up to 2 hidden layers (1024 → 512)."""
    if n_layers < 1 or n_layers > 2:
        raise ValueError(f"MLP2_Variable expects n_layers in [1,2], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP3_Variable(num_classes=10, n_layers=3, **kwargs):
    """MLP with up to 3 hidden layers (1024 → 512 → 256)."""
    if n_layers < 1 or n_layers > 3:
        raise ValueError(f"MLP3_Variable expects n_layers in [1,3], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP8_Variable(num_classes=10, n_layers=8, **kwargs):
    """MLP with up to 8 hidden layers (1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8)."""
    if n_layers < 1 or n_layers > 8:
        raise ValueError(f"MLP8_Variable expects n_layers in [1,8], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP11_Variable(num_classes=10, n_layers=11, **kwargs):
    """MLP with up to 11 hidden layers (1024 → ... → 1)."""
    if n_layers < 1 or n_layers > 11:
        raise ValueError(f"MLP11_Variable expects n_layers in [1,11], got {n_layers}")
    return MLP_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


if __name__ == "__main__":
    # Test the models
    print("Testing MLP Variable with halving architecture:")
    print("=" * 80)
    for n in range(1, 12):
        model = MLP_Variable(n_layers=n)
        x = torch.randn(32, 1, 28, 28)
        y = model(x)
        hidden_dims_str = " → ".join(map(str, model.hidden_dims))
        print(f"MLP with {n:2d} hidden layers: {hidden_dims_str:40s} | params: {model.count_parameters():,}")
