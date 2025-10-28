import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from gwo_benchmark.base import GWOModule

class DeformableConvGWO(GWOModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

        # Offset prediction network
        self.offset_net = nn.Conv2d(in_channels, 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=3, padding=1, stride=stride)

        # The deformable convolution itself
        self.deform_conv = DeformConv2d(in_channels, out_channels, self.kernel_size, stride=stride, padding=int((kernel_size-1)/2))
        self.relu = nn.ReLU()

    @property
    def C_D(self) -> int:
        # CONTENT_AWARE (P) + DENSE_SQUARE(k) (S) + SHARED_KERNEL (W)
        # C_D is higher for CONTENT_AWARE because it requires a separate network.
        return 2 + 1 + 1

    def get_parametric_complexity_modules(self) -> list[nn.Module]:
        # The offset prediction network is the dynamic part
        return [self.offset_net]

    def forward(self, x):
        offsets = self.offset_net(x)
        return self.relu(self.deform_conv(x, offsets))
