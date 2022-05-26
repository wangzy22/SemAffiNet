import torch
import torch.nn as nn


class ResNetShadow(nn.Module):
    def __init__(self, channels):
        super(ResNetShadow, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels, bias=False)
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels, bias=False)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=channels, bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.Parameter(torch.ones_like(m.weight))

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.maxpool0(x0)

        x2 = self.conv2(x0)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        return [x4, x3, x2, x0]
