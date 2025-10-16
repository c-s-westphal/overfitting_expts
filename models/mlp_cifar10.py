"""
Variable-depth MLP models for CIFAR-10 with LayerNorm and residual connections.

Architecture uses LayerNorm (pre-activation), residual connections, and dropout
for better training stability and regularization.
"""
import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with pre-activation LayerNorm.

    Architecture: LayerNorm -> Linear -> ReLU -> Dropout -> residual connection

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


class MLP_CIFAR10(nn.Module):
    """
    Variable-depth MLP for CIFAR-10 classification with LayerNorm and residual connections.

    Architecture:
    - Input: 3072 (32x32x3 flattened)
    - First layer: Linear projection to hidden_dim
    - Hidden layers: Residual MLP blocks with LayerNorm (pre-activation)
    - Output: 10 classes

    Args:
        num_classes (int): Number of output classes (default: 10)
        n_layers (int): Number of hidden layers (1-50)
        hidden_dim (int): Hidden dimension for all layers (default: 256)
        dropout (float): Dropout probability (default: 0.3)
    """
    def __init__(self, num_classes=10, n_layers=3, hidden_dim=256, dropout=0.3):
        super(MLP_CIFAR10, self).__init__()

        if n_layers < 1 or n_layers > 50:
            raise ValueError(f"n_layers must be between 1 and 50, got {n_layers}")

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = 3072  # 32x32x3 CIFAR-10
        self.dropout = dropout

        # First layer: project input to hidden dimension
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.input_relu = nn.ReLU(inplace=True)
        self.input_dropout = nn.Dropout(dropout)

        # Hidden layers with residual connections
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout)
            for _ in range(n_layers - 1)
        ])

        # Final layer norm before output
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)

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
    # Test the models
    print("Testing MLP_CIFAR10 with LayerNorm and residual connections:")
    print("=" * 80)

    for n in range(1, 11):
        model = MLP_CIFAR10(n_layers=n, hidden_dim=256, dropout=0.3)
        x = torch.randn(32, 3, 32, 32)
        y = model(x)

        print(f"MLP with {n:2d} hidden layers: "
              f"hidden_dim={model.hidden_dim:4d} | "
              f"params: {model.count_parameters():,} | "
              f"output shape: {y.shape}")

        # Verify first hidden layer hook
        first_hidden = model.get_first_hidden_layer()
        print(f"  First hidden layer module: {type(first_hidden).__name__}")

    print("=" * 80)
