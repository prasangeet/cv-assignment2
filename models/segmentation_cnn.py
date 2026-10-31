"""Compact U²-Net-inspired encoder-decoder for binary salient-object segmentation."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial convolution followed by pointwise channel mixing."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 3,
                padding=dilation, dilation=dilation,
                groups=in_channels, bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RSUBlock(nn.Module):
    """Small residual U-block that extracts features at two internal scales."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.enc1 = DepthwiseSeparableConv(out_channels, mid_channels)
        self.enc2 = DepthwiseSeparableConv(mid_channels, mid_channels)
        self.center = DepthwiseSeparableConv(mid_channels, mid_channels, dilation=2)
        self.dec2 = DepthwiseSeparableConv(mid_channels * 2, mid_channels)
        self.dec1 = DepthwiseSeparableConv(mid_channels + out_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    @staticmethod
    def upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.project(x)
        e1 = self.enc1(residual)
        e2 = self.enc2(self.pool(e1))
        center = self.center(self.pool(e2))
        d2 = self.dec2(torch.cat((self.upsample(center, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self.upsample(d2, e1), residual), dim=1))
        return d1 + residual


class SegmentationCNN(nn.Module):
    """SwiftMask-inspired lightweight nested residual encoder-decoder.

    Input:  (B, 3, H, W), RGB float image in [0, 1].
    Output: (B, 1, H, W), raw logits (apply sigmoid for probabilities).
    """

    def __init__(self):
        super().__init__()
        self.enc1 = RSUBlock(3, 8, 16)
        self.enc2 = RSUBlock(16, 12, 24)
        self.enc3 = RSUBlock(24, 16, 32)
        self.enc4 = RSUBlock(32, 24, 48)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.center = RSUBlock(48, 32, 64)
        self.dec4 = RSUBlock(64 + 48, 24, 48)
        self.dec3 = RSUBlock(48 + 32, 16, 32)
        self.dec2 = RSUBlock(32 + 24, 12, 24)
        self.dec1 = RSUBlock(24 + 16, 8, 16)
        self.head = nn.Conv2d(16, 1, 1)

    @staticmethod
    def upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat((self.upsample(center, e4), e4), dim=1))
        d3 = self.dec3(torch.cat((self.upsample(d4, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self.upsample(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self.upsample(d2, e1), e1), dim=1))
        logits = self.head(d1)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(
                logits, size=original_size, mode="bilinear", align_corners=False
            )
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
