# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
from models.resnet_mink import ResNetBase
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from models.me_common import ConvType, conv_dw
import MinkowskiEngine as ME

import torch
from torch import nn as nn


class Res16UNetShadow(ResNetBase):
  BLOCK = None
  PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
  DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
  INIT_DIM = 32
  OUT_PIXEL_DIST = 1
  NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, D=3, **kwargs):
    super(Res16UNetShadow, self).__init__(in_channels, in_channels, D)

  def weight_initialization(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiChannelwiseConvolution):
        m.kernel = nn.Parameter(torch.ones_like(m.kernel))

  def network_initialization(self, in_channels, out_channels, D):
    # Setup net_metadata

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    # Output of the first conv concated to conv6
    self.inplanes = self.INIT_DIM

    self.conv1p1s2 = conv_dw(
        in_channels,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

    self.conv2p2s2 = conv_dw(
        in_channels,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

    self.conv3p4s2 = conv_dw(
        in_channels,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

    self.conv4p8s2 = conv_dw(
        in_channels,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

  def forward(self, x):
    out_p1s2 = self.conv1p1s2(x)
    out_p2s2 = self.conv2p2s2(out_p1s2)
    out_p4s2 = self.conv3p4s2(out_p2s2)
    out_p8s2 = self.conv4p8s2(out_p4s2)

    return [out_p8s2, out_p4s2, out_p2s2, out_p1s2]


class Res16UNet14(Res16UNetShadow):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetShadow):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(Res16UNetShadow):
  BLOCK = BasicBlock
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet50(Res16UNetShadow):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetShadow):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
  LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
  PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18E(Res16UNet18):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet18F(Res16UNet18):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34A(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 96, 96)