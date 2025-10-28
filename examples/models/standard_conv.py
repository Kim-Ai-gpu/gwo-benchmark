import torch
import torch.nn as nn
from gwo_benchmark.base import GWOModule

class StandardConvGWO(GWOModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()

    @property
    def C_D(self) -> int:
        # STATIC_SLIDING (P) + DENSE_SQUARE(k) (S) + SHARED_KERNEL (W)
        return 1 + 1 + 1

    def get_parametric_complexity_modules(self) -> list[nn.Module]:
        # Standard convolution has no dynamic parameter network
        return []

    def forward(self, x):
        return self.relu(self.conv(x))
