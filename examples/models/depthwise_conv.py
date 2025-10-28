# examples/models/depthwise_conv.py
import torch
import torch.nn as nn
from gwo_benchmark.base import GWOModule

class DepthwiseSeparableConvGWO(GWOModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=kernel_size//2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()

    @property
    def C_D(self) -> int:
        return 4

    def get_parametric_complexity_modules(self) -> list[nn.Module]:
        return []

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)