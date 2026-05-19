"""VisibilityUNet — per MODEL_SPEC §4.

Lightweight U-Net for predicting the per-pixel shadow factor of the hero
light from a 9-channel input tensor (see `dataset.py` for the channel
layout). Architecture choices are NPU-friendly first: depthwise-separable
decoder, bilinear-resize upsampling (no transposed conv), ReLU6, BN folded
at export. Ops used are within the NNAPI 1.2+ standard set so the
Hexagon / APU 780 / Exynos 1580 delegates have no excuse to fall back to
CPU.

Encoder block (per spec §4):
    Conv3x3 + BN + ReLU6
    DepthwiseConv3x3 + BN + ReLU6
    MaxPool2x2 between stages

Decoder block:
    Bilinear upsample 2x
    Concat skip
    DepthwiseSeparable (Depthwise3x3 + BN + ReLU6 + Pointwise1x1 + BN + ReLU6)

Output: 1x1 conv -> sigmoid.

Channel widths picked to land under the 500K param budget while leaving
the encoder bottleneck wide enough to carry the per-pixel light-relative
features the shadow prediction needs.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_CHANNELS: List[int] = [32, 48, 64, 96]
BOTTLENECK_CHANNELS: int = 128


class EncoderBlock(nn.Module):
    """Conv3x3 + DepthwiseConv3x3, both with BN + ReLU6."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.dw = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,
                            groups=out_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv(x)))
        x = self.act(self.bn2(self.dw(x)))
        return x


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise3x3 + Pointwise1x1, both with BN + ReLU6."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class Bottleneck(nn.Module):
    """Pointwise expand to bottleneck width, then two depthwise blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)
        self.block1 = DepthwiseSeparableBlock(out_ch, out_ch)
        self.block2 = DepthwiseSeparableBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.proj(x)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class VisibilityUNet(nn.Module):
    """9-channel input -> 1-channel shadow factor output. ~250K params."""

    def __init__(self, in_channels: int = 9):
        super().__init__()
        ch = ENCODER_CHANNELS
        bot = BOTTLENECK_CHANNELS

        # Encoder
        self.enc1 = EncoderBlock(in_channels, ch[0])
        self.enc2 = EncoderBlock(ch[0], ch[1])
        self.enc3 = EncoderBlock(ch[1], ch[2])
        self.enc4 = EncoderBlock(ch[2], ch[3])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = Bottleneck(ch[3], bot)

        # Decoder. Each input concat: upsampled (prev_out_ch) + skip (ch[i]).
        self.dec4 = DepthwiseSeparableBlock(bot + ch[3], ch[3])
        self.dec3 = DepthwiseSeparableBlock(ch[3] + ch[2], ch[2])
        self.dec2 = DepthwiseSeparableBlock(ch[2] + ch[1], ch[1])
        self.dec1 = DepthwiseSeparableBlock(ch[1] + ch[0], ch[0])

        # Head
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder + skip capture
        s1 = self.enc1(x)            # [B, 32, 256, 256]
        s2 = self.enc2(self.pool(s1))  # [B, 48, 128, 128]
        s3 = self.enc3(self.pool(s2))  # [B, 64, 64, 64]
        s4 = self.enc4(self.pool(s3))  # [B, 96, 32, 32]

        b = self.bottleneck(self.pool(s4))  # [B, 128, 16, 16]

        # Decoder
        d4 = self.dec4(torch.cat([self._up(b),  s4], dim=1))  # [B, 96, 32, 32]
        d3 = self.dec3(torch.cat([self._up(d4), s3], dim=1))  # [B, 64, 64, 64]
        d2 = self.dec2(torch.cat([self._up(d3), s2], dim=1))  # [B, 48, 128, 128]
        d1 = self.dec1(torch.cat([self._up(d2), s1], dim=1))  # [B, 32, 256, 256]

        return torch.sigmoid(self.head(d1))  # [B, 1, 256, 256]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(in_size: int = 256, in_channels: int = 9) -> int:
    """Hand-rolled MAC counter for VisibilityUNet at HxW = in_size x in_size.

    Counts Conv2D MACs only (BN/ReLU/upsample are negligible). Matches the
    architecture as constructed in `VisibilityUNet.__init__`.
    """
    ch = ENCODER_CHANNELS
    bot = BOTTLENECK_CHANNELS

    def conv2d_macs(c_in: int, c_out: int, k: int, h: int, w: int,
                    groups: int = 1) -> int:
        return (k * k * c_in // groups) * c_out * h * w

    macs = 0
    sizes = [in_size, in_size // 2, in_size // 4, in_size // 8, in_size // 16]
    encoder_ins = [in_channels, ch[0], ch[1], ch[2]]
    encoder_outs = ch

    # Encoder: each block is Conv3x3 (full) + DepthwiseConv3x3. Spatial size is
    # the input size of the block (before the next pool).
    for s, c_in, c_out in zip(sizes[:4], encoder_ins, encoder_outs):
        macs += conv2d_macs(c_in, c_out, 3, s, s)                     # Conv3x3
        macs += conv2d_macs(c_out, c_out, 3, s, s, groups=c_out)      # Depthwise3x3

    # Bottleneck at sizes[4]: 1x1 proj + 2 DSep blocks.
    bs = sizes[4]
    macs += conv2d_macs(ch[3], bot, 1, bs, bs)                        # 1x1 proj
    for _ in range(2):
        macs += conv2d_macs(bot, bot, 3, bs, bs, groups=bot)          # Depthwise3x3
        macs += conv2d_macs(bot, bot, 1, bs, bs)                      # Pointwise1x1

    # Decoder: each DSep block at the post-upsample size, concat skip.
    dec_ins = [bot + ch[3], ch[3] + ch[2], ch[2] + ch[1], ch[1] + ch[0]]
    dec_outs = [ch[3], ch[2], ch[1], ch[0]]
    dec_sizes = [sizes[3], sizes[2], sizes[1], sizes[0]]
    for s, c_in, c_out in zip(dec_sizes, dec_ins, dec_outs):
        macs += conv2d_macs(c_in, c_in, 3, s, s, groups=c_in)         # Depthwise3x3
        macs += conv2d_macs(c_in, c_out, 1, s, s)                     # Pointwise1x1

    # Head 1x1 at full res.
    macs += conv2d_macs(ch[0], 1, 1, in_size, in_size)
    return macs


if __name__ == "__main__":
    m = VisibilityUNet()
    n = count_parameters(m)
    macs = count_macs()
    print(f"VisibilityUNet  params = {n:,}    (budget < 500_000)")
    print(f"                MACs   = {macs / 1e6:.1f}M    (per 256x256 frame)")
    assert n < 500_000, f"param count {n} exceeds budget"
    x = torch.zeros(2, 9, 256, 256)
    y = m(x)
    print(f"input  {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (2, 1, 256, 256), y.shape
