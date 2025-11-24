"""
Variable-depth MLP models for CelebA.

Studies how the number of hidden layers affects generalization
within the same architecture family (fully-connected MLPs).

Architecture uses fixed 512 neurons for all hidden layers.
Supports flexible input sizes (e.g., 64x64x3 for CelebA).
"""
import torch
import torch.nn as nn


class MLP_CelebA(nn.Module):
    """
    Variable-depth MLP for CelebA classification with fixed neurons per layer.

    Architecture:
    - Input: image_size * image_size * 3 (flattened RGB image)
    - Hidden layers: All layers have hidden_dim neurons
    - Optional BatchNorm after each hidden layer
    - Output: num_classes

    Args:
        num_classes (int): Number of output classes (default: 2)
        n_layers (int): Number of hidden layers (1-50)
        hidden_dim (int): Hidden dimension for all layers (default: 512)
        with_bn (bool): Whether to use BatchNorm (default: True)
        dropout_p (float): Dropout probability (default: 0.0)
        image_size (int): Size of input images (default: 64)
    """
    def __init__(self, num_classes=2, n_layers=3, hidden_dim=512, with_bn=True, dropout_p=0.0, image_size=64):
        super(MLP_CelebA, self).__init__()

        if n_layers < 1 or n_layers > 50:
            raise ValueError(f"n_layers must be between 1 and 50, got {n_layers}")

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = image_size * image_size * 3  # RGB image
        self.with_bn = with_bn
        self.dropout_p = dropout_p
        self.image_size = image_size

        # Build the network with fixed hidden dimensions
        layers = []

        # First hidden layer: input_dim -> hidden_dim -> (BN) -> ReLU -> (Dropout)
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        if with_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p > 0:
            layers.append(nn.Dropout(p=dropout_p))

        # Additional hidden layers with fixed dimensions
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if with_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))

        # Output layer: hidden_dim -> num_classes (no BN, ReLU, or Dropout)
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
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Flatten input: (batch, 3, H, W) -> (batch, 3*H*W)
        x = x.view(x.size(0), -1)
        return self.network(x)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def MLP1_CelebA(num_classes=2, n_layers=1, **kwargs):
    """MLP with 1 hidden layer (512 neurons)."""
    if n_layers != 1:
        raise ValueError(f"MLP1_CelebA expects n_layers=1, got {n_layers}")
    return MLP_CelebA(num_classes=num_classes, n_layers=1, **kwargs)


def MLP5_CelebA(num_classes=2, n_layers=5, **kwargs):
    """MLP with up to 5 hidden layers (512 neurons each)."""
    if n_layers < 1 or n_layers > 5:
        raise ValueError(f"MLP5_CelebA expects n_layers in [1,5], got {n_layers}")
    return MLP_CelebA(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP10_CelebA(num_classes=2, n_layers=10, **kwargs):
    """MLP with up to 10 hidden layers (512 neurons each)."""
    if n_layers < 1 or n_layers > 10:
        raise ValueError(f"MLP10_CelebA expects n_layers in [1,10], got {n_layers}")
    return MLP_CelebA(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP15_CelebA(num_classes=2, n_layers=15, **kwargs):
    """MLP with up to 15 hidden layers (512 neurons each)."""
    if n_layers < 1 or n_layers > 15:
        raise ValueError(f"MLP15_CelebA expects n_layers in [1,15], got {n_layers}")
    return MLP_CelebA(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP20_CelebA(num_classes=2, n_layers=20, **kwargs):
    """MLP with up to 20 hidden layers (512 neurons each)."""
    if n_layers < 1 or n_layers > 20:
        raise ValueError(f"MLP20_CelebA expects n_layers in [1,20], got {n_layers}")
    return MLP_CelebA(num_classes=num_classes, n_layers=n_layers, **kwargs)


def MLP25_CelebA(num_classes=2, n_layers=25, **kwargs):
    """MLP with up to 25 hidden layers (512 neurons each)."""
    if n_layers < 1 or n_layers > 25:
        raise ValueError(f"MLP25_CelebA expects n_layers in [1,25], got {n_layers}")
    return MLP_CelebA(num_classes=num_classes, n_layers=n_layers, **kwargs)


if __name__ == "__main__":
    # Test the models
    print("Testing MLP CelebA with fixed 512 neurons:")
    print("=" * 80)
    for n in [1, 5, 10, 15, 20, 25]:
        model = MLP_CelebA(n_layers=n, image_size=64)
        x = torch.randn(32, 3, 64, 64)
        y = model(x)
        hidden_str = f"{model.hidden_dim} × {n}"
        print(f"MLP with {n:2d} hidden layers: {hidden_str:40s} | params: {model.count_parameters():,}")
