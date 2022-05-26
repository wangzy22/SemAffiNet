import torch
from torch import nn
import torch.nn.functional as F

import models.resnet_d as models


class ResUnet(nn.Module):
    def __init__(self, layers=18, classes=2, BatchNorm=nn.BatchNorm2d, pretrained=True, out_channels=128):
        super(ResUnet, self).__init__()
        assert classes > 1
        models.BatchNorm = BatchNorm
        if layers == 18:
            resnet = models.resnet18(pretrained=True, deep_base=False)
            block = models.BasicBlock
            layers = [2, 2, 2, 2]
        elif layers == 34:
            resnet = models.resnet34d(pretrained=True)
            block = models.BasicBlock
            layers = [3, 4, 6, 3]
        elif layers == 50:
            resnet = models.resnet50(pretrained=True, deep_base=False)
            block = models.BasicBlock
            layers = [3, 4, 6, 3]
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)

        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.relu = nn.ReLU()

        # Decoder
        # self.up4 = nn.Sequential(nn.ConvTranspose2d(512,256,kernel_size=2,stride=2),BatchNorm(256),nn.ReLU())
        # self.up4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), BatchNorm(256), nn.ReLU())
        self.class_convfeat = nn.Conv2d(512, out_channels, kernel_size=1)
        self.up4 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.class_conv4 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(out_channels, affine=False)
        inplanes = out_channels + 256
        self.delayer4, _ = models.make_blocks(block, [256], [layers[-1]], inplanes)
        self.delayer4 = self.delayer4[0][1]

        # self.up3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), BatchNorm(128), nn.ReLU())
        self.up3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.class_conv3 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels, affine=False)
        inplanes = out_channels + 128
        self.delayer3, _ = models.make_blocks(block, [128], [layers[-2]], inplanes)
        self.delayer3 = self.delayer3[0][1]

        # self.up2 = nn.Sequential(nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1), BatchNorm(96), nn.ReLU())
        self.up2 = nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)
        self.class_conv2 = nn.Conv2d(96, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=False)
        inplanes = out_channels + 64
        self.delayer2, _ = models.make_blocks(block, [96], [layers[-3]], inplanes)
        self.delayer2 = self.delayer2[0][1]

        self.cls = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     BatchNorm(256), nn.ReLU(inplace=True),
                                     nn.Conv2d(256, classes, kernel_size=1))

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.layer0(x)  # 1/4
        x2 = self.layer1(x)  # 1/4
        x3 = self.layer2(x2)  # 1/8
        x4 = self.layer3(x3)  # 1/16
        x5 = self.layer4(x4)  # 1/32
        p4 = self.up4(F.interpolate(x5, x4.shape[-2:], mode='bilinear', align_corners=True))
        p4 = torch.cat([p4, x4], dim=1)
        p4 = self.delayer4(p4)
        p3 = self.up3(F.interpolate(p4, x3.shape[-2:], mode='bilinear', align_corners=True))
        p3 = torch.cat([p3, x3], dim=1)
        p3 = self.delayer3(p3)
        p2 = self.up2(F.interpolate(p3, x2.shape[-2:], mode='bilinear', align_corners=True))
        p2 = torch.cat([p2, x2], dim=1)
        p2 = self.delayer2(p2)
        x = self.cls(p2)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:
            aux = self.aux(x4)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            return x, aux
        else:
            return x
