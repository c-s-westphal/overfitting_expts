"""
Variable-depth All-Conv models for CIFAR-10.

Studies how the number of convolutional layers affects generalization
within a simple all-convolutional architecture with fixed width.
"""
import torch
import torch.nn as nn


class AllConv_Variable(nn.Module):
    """
    Variable-depth All-Convolutional network for CIFAR-10 classification.

    Architecture:
    - Input: 3×32×32 (CIFAR-10)
    - Conv layers: n_layers × conv3×3(num_channels) with ReLU
    - All conv layers maintain 32×32 spatial size (padding=1, stride=1)
    - AdaptiveAvgPool to 4×4
    - Classifier: num_channels×16 → num_classes

    Args:
        num_classes (int): Number of output classes (default: 10)
        n_layers (int): Number of convolutional layers (1-5)
        num_channels (int): Number of channels in conv layers (default: 128)
    """
    def __init__(self, num_classes=10, n_layers=3, num_channels=128):
        super(AllConv_Variable, self).__init__()

        if n_layers < 1 or n_layers > 15:
            raise ValueError(f"n_layers must be between 1 and 15, got {n_layers}")

        self.n_layers = n_layers
        self.num_channels = num_channels
        self.num_classes = num_classes

        # Build the convolutional layers
        conv_layers = []

        # First conv layer: 3 → num_channels
        conv_layers.append(nn.Conv2d(3, num_channels, kernel_size=3, padding=1, stride=1, bias=True))
        conv_layers.append(nn.ReLU(inplace=True))

        # Additional conv layers: num_channels → num_channels
        for _ in range(n_layers - 1):
            conv_layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=True))
            conv_layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*conv_layers)

        # Adaptive pooling to 4×4
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier: num_channels×4×4 → num_classes
        self.classifier = nn.Linear(num_channels * 4 * 4, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming (He) initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Conv layers (maintain 32×32)
        x = self.features(x)
        # Pool to 4×4
        x = self.pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Classify
        x = self.classifier(x)
        return x

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def AllConv1_Variable(num_classes=10, n_layers=1, **kwargs):
    """All-Conv with 1 conv layer (variable depth 1-1)."""
    if n_layers != 1:
        raise ValueError(f"AllConv1_Variable expects n_layers=1, got {n_layers}")
    return AllConv_Variable(num_classes=num_classes, n_layers=1, **kwargs)


def AllConv2_Variable(num_classes=10, n_layers=2, **kwargs):
    """All-Conv with up to 2 conv layers (variable depth 1-2)."""
    if n_layers < 1 or n_layers > 2:
        raise ValueError(f"AllConv2_Variable expects n_layers in [1,2], got {n_layers}")
    return AllConv_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def AllConv3_Variable(num_classes=10, n_layers=3, **kwargs):
    """All-Conv with up to 3 conv layers (variable depth 1-3)."""
    if n_layers < 1 or n_layers > 3:
        raise ValueError(f"AllConv3_Variable expects n_layers in [1,3], got {n_layers}")
    return AllConv_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def AllConv4_Variable(num_classes=10, n_layers=4, **kwargs):
    """All-Conv with up to 4 conv layers (variable depth 1-4)."""
    if n_layers < 1 or n_layers > 4:
        raise ValueError(f"AllConv4_Variable expects n_layers in [1,4], got {n_layers}")
    return AllConv_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


def AllConv5_Variable(num_classes=10, n_layers=5, **kwargs):
    """All-Conv with up to 5 conv layers (variable depth 1-5)."""
    if n_layers < 1 or n_layers > 5:
        raise ValueError(f"AllConv5_Variable expects n_layers in [1,5], got {n_layers}")
    return AllConv_Variable(num_classes=num_classes, n_layers=n_layers, **kwargs)


if __name__ == "__main__":
    # Test the models
    for n in range(1, 6):
        model = AllConv_Variable(n_layers=n)
        x = torch.randn(32, 3, 32, 32)
        y = model(x)
        print(f"All-Conv with {n} layers: input {x.shape} -> output {y.shape}, params: {model.count_parameters():,}")

        # Print feature map sizes
        with torch.no_grad():
            feat = model.features(x)
            pooled = model.pool(feat)
            print(f"  After conv: {feat.shape}, After pool: {pooled.shape}, Flattened: {pooled.view(pooled.size(0), -1).shape}")
