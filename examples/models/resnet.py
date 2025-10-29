# 파일 위치: examples/models/resnet.py

import torch
import torch.nn as nn
from gwo_benchmark.base import GWOModule

class ConvGWO(GWOModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    @property
    def C_D(self) -> int:
        # STATIC_SLIDING (P) + DENSE_SQUARE(k) (S) + SHARED_KERNEL (W)
        return 1 + 1 + 1

    def get_parametric_complexity_modules(self) -> list[nn.Module]:
        return []

    def forward(self, x):
        return self.conv(x)

class BasicBlockGWO(GWOModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = ConvGWO(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ConvGWO(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_conv = ConvGWO(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)
            self.shortcut = nn.Sequential(self.shortcut_conv, self.shortcut_bn)

    @property
    def C_D(self) -> int:
        cd_sum = self.conv1.C_D + self.conv2.C_D
        if hasattr(self, 'shortcut_conv'):
            cd_sum += self.shortcut_conv.C_D
        return cd_sum

    def get_parametric_complexity_modules(self) -> list[nn.Module]:
        return []

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetGWO(GWOModule):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = ConvGWO(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.gwo_modules = [self.conv1]
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in layer:
                self.gwo_modules.append(b)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @property
    def C_D(self) -> int:
        return sum(m.C_D for m in self.gwo_modules)

    def get_parametric_complexity_modules(self) -> list[nn.Module]:
        all_modules = []
        for m in self.gwo_modules:
            all_modules.extend(m.get_parametric_complexity_modules())
        return all_modules

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet18GWO():
    return ResNetGWO(BasicBlockGWO, [2, 2, 2, 2])