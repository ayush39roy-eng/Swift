"""ResNet-18 architecture used as the victim / backdoored classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """Standard ResNet-18 for 32x32 inputs (CIFAR-style stem)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        """Penultimate-layer (pre-fc) feature vector, used for Wasserstein geometry."""
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))
