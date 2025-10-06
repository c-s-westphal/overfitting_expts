import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, depth, num_classes=10, block_type='basic'):
        super(PreActResNet, self).__init__()
        assert block_type in ('basic', 'bottleneck')

        if block_type == 'basic':
            # CIFAR PreAct with BasicBlock has depth = 6n + 2
            assert (depth - 2) % 6 == 0, "For basic block, depth must be 6n+2"
            n = (depth - 2) // 6
            block = PreActBasicBlock
            widths = [16, 32, 64]
        else:
            # CIFAR PreAct with Bottleneck has depth = 9n + 2
            assert (depth - 2) % 9 == 0, "For bottleneck block, depth must be 9n+2"
            n = (depth - 2) // 9
            block = PreActBottleneck
            widths = [16, 32, 64]

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, widths[0], n, stride=1)
        self.layer2 = self._make_layer(block, widths[1], n, stride=2)
        self.layer3 = self._make_layer(block, widths[2], n, stride=2)
        self.bn = nn.BatchNorm2d(widths[2] * block.expansion)
        self.linear = nn.Linear(widths[2] * block.expansion, num_classes)

        # Kaiming initialization consistent with common CIFAR implementations
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def build_preact_resnet(depth, num_classes=10, prefer_stable=True):
    """Builds a PreAct ResNet for CIFAR at a given depth.

    prefer_stable=True chooses bottleneck for very deep nets when depth fits 9n+2,
    otherwise falls back to basic which is stable for standard CIFAR depths.
    """
    if prefer_stable and depth >= 164:
        # Try bottleneck if depth is valid for 9n+2; else use basic
        if (depth - 2) % 9 == 0:
            return PreActResNet(depth, num_classes=num_classes, block_type='bottleneck')
    return PreActResNet(depth, num_classes=num_classes, block_type='basic')


# Convenience constructors for requested depths
def PreActResNet20():
    return build_preact_resnet(20)


def PreActResNet32():
    return build_preact_resnet(32)


def PreActResNet44():
    return build_preact_resnet(44)


def PreActResNet56():
    return build_preact_resnet(56)


def PreActResNet80():
    return build_preact_resnet(80)


def PreActResNet110():
    return build_preact_resnet(110)


def PreActResNet218():
    return build_preact_resnet(218)


