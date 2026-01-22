import torch
import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class SRGenerator(nn.Module):
    """
    基于 SRResNet 的生成器
    对应报告文献 [42] SRGAN
    """

    def __init__(self, scale_factor=4, residual_blocks=16):
        super(SRGenerator, self).__init__()
        # CT是单通道灰度图，所以 input_channels=1
        self.conv_input = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()

        # 残差块堆叠
        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(residual_blocks)])

        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(64)

        # 上采样模块 (PixelShuffle)
        upsample_layers = []
        for _ in range(int(math.log2(scale_factor))):
            upsample_layers.append(nn.Conv2d(64, 256, kernel_size=3, padding=1))
            upsample_layers.append(nn.PixelShuffle(upscale_factor=2))
            upsample_layers.append(nn.PReLU())
        self.upsample = nn.Sequential(*upsample_layers)

        self.conv_output = nn.Conv2d(64, 1, kernel_size=9, padding=4)
        # 最后输出用 Tanh 还是不激活取决于数据归一化，这里CT归一化到0-1，可以用Sigmoid或直接输出
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv_input(x)
        x1 = self.prelu(x1)

        res = self.residuals(x1)
        res = self.conv_mid(res)
        res = self.bn_mid(res)

        x = x1 + res  # Skip connection
        x = self.upsample(x)
        x = self.conv_output(x)
        return self.activation(x)


class SRDiscriminator(nn.Module):
    """
    判别器：判断输入的CT切片是真实的还是生成的
    """

    def __init__(self):
        super(SRDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            self._block(64, 64, stride=2),
            self._block(64, 128, stride=1),
            self._block(128, 128, stride=2),
            self._block(128, 256, stride=1),
            self._block(256, 256, stride=2),
            self._block(256, 512, stride=1),
            self._block(512, 512, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            # 输出 logits，在 Loss 函数中处理 sigmoid
        )

    def _block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classifier(features)