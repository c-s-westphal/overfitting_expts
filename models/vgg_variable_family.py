"""
Variable-depth VGG models for CIFAR-10.

Uses the first N convolutional layers from standard VGG architectures,
followed by adaptive pooling and a unified classifier head.
This enables studying how network depth affects generalization.
"""
import torch
import torch.nn as nn


# Full VGG configurations (no BatchNorm variant) for CIFAR-10
# Limited to 2 maxpools to end at 8x8 spatial features
VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 512, 512, 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 512, 512, 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}

VGG_MAX_LAYERS = {
    'VGG11': 8,
    'VGG13': 10,
    'VGG16': 13,
    'VGG19': 16,
}


class _VGGVariableBase(nn.Module):
    """
    Variable-depth VGG base class.

    Truncates a full VGG configuration to the first n_layers convolutional layers,
    followed by adaptive pooling and a unified MLP classifier.

    Args:
        num_classes: Number of output classes
        n_layers: Number of convolutional layers to use (must be >= 4)
        full_cfg: Full VGG configuration (list of channels and 'M' for maxpool)
        arch_name: Architecture name for validation (e.g., 'VGG11')
        max_layers: Maximum number of conv layers in this architecture
    """

    def __init__(self, num_classes, n_layers, full_cfg, arch_name, max_layers, with_bn=False, dropout_p=0.5):
        super(_VGGVariableBase, self).__init__()

        if n_layers < 4:
            raise ValueError(f"{arch_name}: Minimum 4 layers required, got {n_layers}")
        if n_layers > max_layers:
            raise ValueError(f"{arch_name}: Maximum {max_layers} layers, got {n_layers}")

        self.arch_name = arch_name
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.with_bn = with_bn
        self.dropout_p = dropout_p

        # Truncate config to n_layers convolutional layers
        truncated_cfg = self._get_truncated_cfg(full_cfg, n_layers)

        # Build feature extractor and get last channel count
        self.features, last_channels = self._make_layers(truncated_cfg)

        # Adaptive pooling to 1x1 spatial size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ADAPTIVE CLASSIFIER: Always expand by 2× to ensure gradient flow
        # For 256-ch: 256→512→256→10 (unchanged)
        # For 512-ch: 512→1024→256→10 (fixes bottleneck)
        hidden_dim = max(512, last_channels * 2)
        self.classifier = nn.Sequential(
            nn.Linear(last_channels, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes)
        )

        # Weight initialization tailored for ReLU networks
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_truncated_cfg(self, full_cfg, n_layers):
        """
        Truncate configuration to first n_layers convolutional layers.

        Counts only convolutional layers (integers in config).
        Stops after n_layers conv layers, includes preceding maxpools.

        Example: VGG11 config = [64, 'M', 128, 'M', 256, 256, 'M', ...]
        - n_layers=4 → [64, 'M', 128, 'M', 256, 256]
        """
        conv_count = 0
        truncated_cfg = []

        for item in full_cfg:
            if isinstance(item, int):
                conv_count += 1
                truncated_cfg.append(item)
                if conv_count == n_layers:
                    break
            else:
                # Include pooling layers
                truncated_cfg.append(item)

        return truncated_cfg

    def _make_layers(self, cfg):
        """
        Build convolutional layers from config (no BatchNorm).

        Returns:
            nn.Sequential: Feature extraction layers
            int: Number of output channels from last conv layer
        """
        layers = []
        in_channels = 3
        last_channels = 3

        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # Use bias only when BatchNorm is not present
                conv = nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=not self.with_bn)
                layers.append(conv)
                if self.with_bn:
                    layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
                last_channels = x

        return nn.Sequential(*layers), last_channels

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Factory functions for each VGG variant
def VGG11_Variable(num_classes=10, n_layers=8, with_bn=True, dropout_p=0.5):
    """VGG11 Variable: 4-8 convolutional layers with unified classifier."""
    return _VGGVariableBase(
        num_classes=num_classes,
        n_layers=n_layers,
        full_cfg=VGG_CONFIGS['VGG11'],
        arch_name='VGG11',
        max_layers=VGG_MAX_LAYERS['VGG11'],
        with_bn=with_bn,
        dropout_p=dropout_p
    )


def VGG13_Variable(num_classes=10, n_layers=10, with_bn=True, dropout_p=0.5):
    """VGG13 Variable: 4-10 convolutional layers with unified classifier."""
    return _VGGVariableBase(
        num_classes=num_classes,
        n_layers=n_layers,
        full_cfg=VGG_CONFIGS['VGG13'],
        arch_name='VGG13',
        max_layers=VGG_MAX_LAYERS['VGG13'],
        with_bn=with_bn,
        dropout_p=dropout_p
    )


def VGG16_Variable(num_classes=10, n_layers=13, with_bn=True, dropout_p=0.5):
    """VGG16 Variable: 4-13 convolutional layers with unified classifier."""
    return _VGGVariableBase(
        num_classes=num_classes,
        n_layers=n_layers,
        full_cfg=VGG_CONFIGS['VGG16'],
        arch_name='VGG16',
        max_layers=VGG_MAX_LAYERS['VGG16'],
        with_bn=with_bn,
        dropout_p=dropout_p
    )


def VGG19_Variable(num_classes=10, n_layers=16, with_bn=True, dropout_p=0.5):
    """VGG19 Variable: 4-16 convolutional layers with unified classifier."""
    return _VGGVariableBase(
        num_classes=num_classes,
        n_layers=n_layers,
        full_cfg=VGG_CONFIGS['VGG19'],
        arch_name='VGG19',
        max_layers=VGG_MAX_LAYERS['VGG19'],
        with_bn=with_bn,
        dropout_p=dropout_p
    )
