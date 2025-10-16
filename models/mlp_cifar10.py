"""
Variable-depth MLP models for CIFAR-10 with LayerNorm and residual connections.

Architecture uses LayerNorm (pre-activation), residual connections, and dropout
for better training stability and regularization.
"""
import torch
import torch.nn as nn
import numpy as np


def compute_smooth_exponential_widths(n_layers, initial_width=1024, target_width=64):
    """Compute layer widths using smooth exponential decay.

    Formula: width[i] = round(initial_width × (target_width / initial_width)^(i / (n_layers - 1)))

    Args:
        n_layers: Number of hidden layers
        initial_width: Width of first hidden layer
        target_width: Width of final hidden layer

    Returns:
        List of hidden layer widths
    """
    if n_layers == 1:
        return [initial_width]

    widths = []
    ratio = target_width / initial_width
    for i in range(n_layers):
        exponent = i / (n_layers - 1)
        width = initial_width * (ratio ** exponent)
        widths.append(int(round(width)))

    return widths


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with pre-activation LayerNorm and optional projection.

    Architecture: LayerNorm -> Linear -> ReLU -> Dropout -> (projection) + residual

    Args:
        in_dim (int): Input dimension size
        out_dim (int): Output dimension size
        dropout (float): Dropout probability
    """
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super(ResidualMLPBlock, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions don't match
        self.projection = None
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        identity = x
        out = self.norm(x)
        out = self.linear(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Apply projection to identity if needed
        if self.projection is not None:
            identity = self.projection(identity)

        out = out + identity  # Residual connection
        return out


class MLP_CIFAR10(nn.Module):
    """
    Variable-depth MLP for CIFAR-10 classification with LayerNorm and residual connections.

    Architecture:
    - Input: 3072 (32x32x3 flattened)
    - First layer: Linear projection to first hidden dimension
    - Hidden layers: Residual MLP blocks with LayerNorm (pre-activation)
    - Output: 10 classes

    Supports variable width per layer using smooth exponential decay.

    Args:
        num_classes (int): Number of output classes (default: 10)
        n_layers (int): Number of hidden layers (1-50)
        hidden_dims (list or int): Hidden dimensions (list for variable, int for fixed)
        dropout (float): Dropout probability (default: 0.3)
        initial_width (int): Initial width for smooth exponential decay (default: 1024)
        target_width (int): Target width for smooth exponential decay (default: 64)
    """
    def __init__(self, num_classes=10, n_layers=3, hidden_dims=None, dropout=0.3,
                 initial_width=1024, target_width=64):
        super(MLP_CIFAR10, self).__init__()

        if n_layers < 1 or n_layers > 50:
            raise ValueError(f"n_layers must be between 1 and 50, got {n_layers}")

        self.n_layers = n_layers
        self.input_dim = 3072  # 32x32x3 CIFAR-10
        self.dropout = dropout

        # Compute hidden dimensions
        if hidden_dims is None:
            # Use smooth exponential decay
            self.hidden_dims = compute_smooth_exponential_widths(n_layers, initial_width, target_width)
        elif isinstance(hidden_dims, int):
            # Fixed width for all layers
            self.hidden_dims = [hidden_dims] * n_layers
        else:
            # Use provided list
            self.hidden_dims = hidden_dims

        if len(self.hidden_dims) != n_layers:
            raise ValueError(f"hidden_dims length ({len(self.hidden_dims)}) must match n_layers ({n_layers})")

        # For backwards compatibility, store first hidden dim as hidden_dim
        self.hidden_dim = self.hidden_dims[0]

        # First layer: project input to first hidden dimension
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.input_relu = nn.ReLU(inplace=True)
        self.input_dropout = nn.Dropout(dropout)

        # Hidden layers with residual connections and variable widths
        self.blocks = nn.ModuleList()
        for i in range(n_layers - 1):
            in_dim = self.hidden_dims[i]
            out_dim = self.hidden_dims[i + 1]
            self.blocks.append(ResidualMLPBlock(in_dim, out_dim, dropout))

        # Final layer norm before output (uses last hidden dim)
        self.final_norm = nn.LayerNorm(self.hidden_dims[-1])

        # Output layer (from last hidden dim to num_classes)
        self.output = nn.Linear(self.hidden_dims[-1], num_classes)

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
            x: Input tensor of shape (batch, 3, 32, 32)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Flatten input: (batch, 3, 32, 32) -> (batch, 3072)
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
        """Get the first hidden layer output (after input projection and activation).

        This is used for masking and MI evaluation.

        Returns:
            The module after which to hook for first hidden layer output.
            Returns the input_dropout module.
        """
        return self.input_dropout

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models with smooth exponential decay
    print("Testing MLP_CIFAR10 with smooth exponential decay (1024 -> 64):")
    print("=" * 80)

    test_depths = [1, 5, 10, 15, 20]
    for n in test_depths:
        model = MLP_CIFAR10(n_layers=n, dropout=0.3, initial_width=1024, target_width=64)
        x = torch.randn(32, 3, 32, 32)
        y = model(x)

        widths_str = " -> ".join([str(w) for w in model.hidden_dims])
        print(f"\nMLP with {n:2d} layers:")
        print(f"  Widths: {widths_str}")
        print(f"  Total params: {model.count_parameters():,}")
        print(f"  Output shape: {y.shape}")

        # Verify first hidden layer hook
        first_hidden = model.get_first_hidden_layer()
        print(f"  First hidden layer module: {type(first_hidden).__name__}, width={model.hidden_dim}")

    print("\n" + "=" * 80)
    print("Testing backwards compatibility with fixed width:")
    print("=" * 80)

    model_fixed = MLP_CIFAR10(n_layers=3, hidden_dims=256, dropout=0.3)
    x = torch.randn(32, 3, 32, 32)
    y = model_fixed(x)
    widths_str = " -> ".join([str(w) for w in model_fixed.hidden_dims])
    print(f"MLP with 3 layers (fixed width 256): {widths_str}")
    print(f"Total params: {model_fixed.count_parameters():,}")

    print("=" * 80)
