"""Globally-blended sinusoidal trigger.

Unlike a localized patch trigger, this pattern is blended at low amplitude
across the entire image, which makes it substantially harder for defenses
such as Neural Cleanse to reverse-engineer (they are designed to find a
small, spatially-isolated trigger region).
"""

import math
import torch


def make_pattern(height=32, width=32, num_channels=3):
    """Fixed sinusoidal pattern, spatial frequency 1/8 cycles/pixel."""
    pattern = torch.zeros(1, num_channels, height, width)
    for c in range(num_channels):
        for i in range(height):
            for j in range(width):
                pattern[0, c, i, j] = (
                    0.5 * math.sin(2 * math.pi * i / 8 + c * math.pi / 3)
                    + 0.5 * math.sin(2 * math.pi * j / 8 + c * math.pi / 3)
                ) * 0.5 + 0.5
    return pattern


class Trigger:
    """Callable trigger: T(x) = (1 - blend_alpha) * x + blend_alpha * pattern."""

    def __init__(self, height=32, width=32, num_channels=3, blend_alpha=0.15, device="cpu"):
        self.blend_alpha = blend_alpha
        self.pattern = make_pattern(height, width, num_channels).to(device)

    def __call__(self, x):
        return (1 - self.blend_alpha) * x + self.blend_alpha * self.pattern.expand_as(x)
