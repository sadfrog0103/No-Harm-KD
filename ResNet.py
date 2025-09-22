import torch
import torch.nn as nn
from typing import Optional, Type, List

# -------------------------
# BasicBlock / Bottleneck
# -------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 正确的 BN 链接：conv1 → bn1 → relu
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        width = out_channels

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# -------------------------
# ResNet backbone
# -------------------------
class ResNet(nn.Module):
    def __init__(self,
                 block: Type[nn.Module],
                 layers: List[int],
                 num_classes: int = 1000,
                 num_channels: int = 3,
                 small_input: bool = False,
                 zero_init_residual: bool = True):
        """
        small_input=True: 为 CIFAR10/100 等小图（32×32）使用 3x3/stride=1 的 stem，并去掉 maxpool
        zero_init_residual=True: 将每个残差分支的最后一个 BN 的权重置 0（更稳的训练）
        """
        super().__init__()
        self.in_channels = 64

        if small_input:
            # CIFAR 风格 stem
            self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.max_pool = nn.Identity()
        else:
            # ImageNet 风格 stem
            self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(zero_init_residual=zero_init_residual)

    def _init_weights(self, zero_init_residual: bool = True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if zero_init_residual:
            # 更稳的起始：把残差分支最后一个 BN 的 gamma 置 0
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.batch_norm2.weight)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# -------------------------
# Factory
# -------------------------
def get_resnet(model_name: str, num_classes: int, channels: int = 3, small_input: bool = False):
    name = model_name.lower()
    if name == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, channels, small_input)
    elif name == "resnet34":
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, channels, small_input)
    elif name == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels, small_input)
    elif name == "resnet101":
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels, small_input)
    elif name == "resnet152":
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels, small_input)
    else:
        raise ValueError(f"Unknown model: {model_name}")
