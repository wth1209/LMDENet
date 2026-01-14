"""
Bi-Focus module
"""

import torch
import torch.nn as nn

from ..modules.conv import Conv
from ..modules.block import Bottleneck


class C2f_BiFocus(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

        self.bifocus = BiFocus(c2, c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))

        return self.bifocus(y)

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class BiFocus(nn.Module):
    # c1: 输入通道数
    # c2: 输出通道数
    def __init__(self, c1, c2):
        super().__init__()
        # 创建一个 水平方向 Focus 模块，它是对输入进行沿 H 方向切分并卷积提取特征。
        # 输入和输出通道数均为 c1。
        # 卷积核大小为 3，步幅为 1。
        self.focus_h = FocusH(c1, c1, 3, 1)
        # 创建一个 垂直方向 Focus 模块，类似于 FocusH，但方向变成 W。
        # 用于提取 水平方向的结构信息。
        self.focus_v = FocusV(c1, c1, 3, 1)
        # 深度可分离卷积模块：
        # 输入通道是 3 * c1，因为输入通道与两个方向特征拼接后会有 3 倍通道数。
        # 输出通道数为 c2，为最终输出目标维度。
        # 卷积核大小为 3×3。
        self.depth_wise = DepthWiseConv(3 * c1, c2, 3)

    def forward(self, x):
        return self.depth_wise(torch.cat([x, self.focus_h(x), self.focus_v(x)], dim=1))


class FocusH(nn.Module):

    def __init__(self, c1, c2, kernel=3, stride=1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)

    def forward(self, x):
        b, _, h, w = x.shape
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, self.c2, h, w // 2], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, self.c2, h, w // 2], device=x.device, dtype=x.dtype)

        x1[..., ::2, :], x1[..., 1::2, :] = x[..., ::2, ::2], x[..., 1::2, 1::2]
        x2[..., ::2, :], x2[..., 1::2, :] = x[..., ::2, 1::2], x[..., 1::2, ::2]

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        result[..., ::2, ::2] = x1[..., ::2, :]
        result[..., 1::2, 1::2] = x1[..., 1::2, :]
        result[..., ::2, 1::2] = x2[..., ::2, :]
        result[..., 1::2, ::2] = x2[..., 1::2, :]

        return result


class FocusV(nn.Module):

    def __init__(self, c1, c2, kernel=3, stride=1):
        super().__init__()
        self.c2 = c2
        self.conv1 = Conv(c1, c2, kernel, stride)
        self.conv2 = Conv(c1, c2, kernel, stride)

    def forward(self, x):
        b, _, h, w = x.shape
        result = torch.zeros(size=[b, self.c2, h, w], device=x.device, dtype=x.dtype)
        x1 = torch.zeros(size=[b, self.c2, h // 2, w], device=x.device, dtype=x.dtype)
        x2 = torch.zeros(size=[b, self.c2, h // 2, w], device=x.device, dtype=x.dtype)

        x1[..., ::2], x1[..., 1::2] = x[..., ::2, ::2], x[..., 1::2, 1::2]
        x2[..., ::2], x2[..., 1::2] = x[..., 1::2, ::2], x[..., ::2, 1::2]

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        result[..., ::2, ::2] = x1[..., ::2]
        result[..., 1::2, 1::2] = x1[..., 1::2]
        result[..., 1::2, ::2] = x2[..., ::2]
        result[..., ::2, 1::2] = x2[..., 1::2]

        return result


class DepthWiseConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = Conv(in_channel, in_channel, kernel, 1, 1, in_channel)
        self.point_conv = Conv(in_channel, out_channel, 1, 1, 0, 1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)

        return out
