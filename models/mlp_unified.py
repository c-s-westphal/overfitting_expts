"""
Unified variable-depth MLP model for CIFAR-10 and MNIST with LayerNorm and residual connections.

Architecture uses LayerNorm (pre-activation), residual connections, and dropout
for better training stability and regularization.
"""
import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """MLP block with pre-activation LayerNorm (no residual connection).

    Architecture: LayerNorm -> Linear -> ReLU -> Dropout

    Args:
        hidden_dim (int): Hidden dimension size
        dropout (float): Dropout probability
    """
    def __init__(self, hidden_dim, dropout=0.3):
        super(ResidualMLPBlock, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.norm(x)
        out = self.linear(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + identity  # Residual connection
        return out


class MLP_Unified(nn.Module):
    """
    Unified variable-depth MLP for CIFAR-10 and MNIST classification with LayerNorm and residual connections.

    Architecture:
    - Input: Auto-detected (3072 for CIFAR-10, 784 for MNIST)
    - First layer: Linear projection to hidden_dim
    - Hidden layers: Residual MLP blocks with LayerNorm (pre-activation)
    - Output: num_classes

    Args:
        num_classes (int): Number of output classes (default: 10)
        n_layers (int): Number of hidden layers (1-50)
        hidden_dim (int): Hidden dimension for all layers (default: 256)
        dropout (float): Dropout probability (default: 0.3)
        input_dim (int): Input dimension. If None, auto-detected from first forward pass
    """
    def __init__(self, num_classes=10, n_layers=3, hidden_dim=256, dropout=0.3, input_dim=None):
        super(MLP_Unified, self).__init__()

        if n_layers < 1 or n_layers > 50:
            raise ValueError(f"n_layers must be between 1 and 50, got {n_layers}")

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self._input_dim = input_dim

        # If input_dim is provided, build the model immediately
        if input_dim is not None:
            self._build_model(input_dim)

    def _build_model(self, input_dim):
        """Build the model layers given input dimension."""
        self._input_dim = input_dim

        # First layer: project input to hidden dimension
        self.input_proj = nn.Linear(self._input_dim, self.hidden_dim)
        self.input_relu = nn.ReLU(inplace=True)
        self.input_dropout = nn.Dropout(self.dropout)

        # Hidden layers with residual connections
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(self.hidden_dim, self.dropout)
            for _ in range(self.n_layers - 1)
        ])

        # Final layer norm before output
        self.final_norm = nn.LayerNorm(self.hidden_dim)

        # Output layer
        self.output = nn.Linear(self.hidden_dim, self.num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming (He) initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, C, H, W)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Auto-detect input dimension if not already set
        if self._input_dim is None:
            # Flatten and detect dimension
            x_flat = x.view(x.size(0), -1)
            input_dim = x_flat.size(1)
            self._build_model(input_dim)
            # Move to same device as input
            self.to(x.device)

        # Flatten input: (batch, C, H, W) -> (batch, input_dim)
        x = x.view(x.size(0), -1)

        # First layer
        x = self.input_proj(x)
        x = self.input_relu(x)
        x = self.input_dropout(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Output
        x = self.output(x)

        return x

    def get_first_hidden_layer(self):
        """Get the first hidden layer output (after first residual block).

        This is used for masking and MI evaluation.

        Returns:
            The module after which to hook for first hidden layer output.
            Returns blocks[0] (requires n_layers >= 2).

        Raises:
            ValueError: If n_layers < 2 (no residual blocks exist).
        """
        if len(self.blocks) > 0:
            return self.blocks[0]
        else:
            raise ValueError(f"Masking requires n_layers >= 2 to mask after first residual block. "
                           f"Got n_layers={self.n_layers}")

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the unified model with both CIFAR-10 and MNIST
    print("Testing MLP_Unified with CIFAR-10 and MNIST:")
    print("=" * 80)

    # Test with CIFAR-10 input
    print("\nCIFAR-10 (3x32x32 = 3072 dims):")
    for n in [1, 5, 10]:
        model = MLP_Unified(num_classes=10, n_layers=n, hidden_dim=256, dropout=0.3, input_dim=3072)
        x = torch.randn(32, 3, 32, 32)
        y = model(x)

        print(f"  MLP with {n:2d} hidden layers: "
              f"params: {model.count_parameters():,} | "
              f"output shape: {y.shape}")

    # Test with MNIST input
    print("\nMNIST (1x28x28 = 784 dims):")
    for n in [1, 5, 10]:
        model = MLP_Unified(num_classes=10, n_layers=n, hidden_dim=256, dropout=0.3, input_dim=784)
        x = torch.randn(32, 1, 28, 28)
        y = model(x)

        print(f"  MLP with {n:2d} hidden layers: "
              f"params: {model.count_parameters():,} | "
              f"output shape: {y.shape}")

    # Test auto-detection
    print("\nAuto-detection test:")
    model = MLP_Unified(num_classes=10, n_layers=3, hidden_dim=256, dropout=0.3)
    x_cifar = torch.randn(16, 3, 32, 32)
    y_cifar = model(x_cifar)
    print(f"  Auto-detected CIFAR-10 input: {model._input_dim} dims, output: {y_cifar.shape}")

    model = MLP_Unified(num_classes=10, n_layers=3, hidden_dim=256, dropout=0.3)
    x_mnist = torch.randn(16, 1, 28, 28)
    y_mnist = model(x_mnist)
    print(f"  Auto-detected MNIST input: {model._input_dim} dims, output: {y_mnist.shape}")

    print("=" * 80)
