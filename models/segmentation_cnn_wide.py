"""Wider hybrid residual U-Net for binary salient-object segmentation."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class StandardConv(nn.Module):
    """Standard 3x3 convolution for high-resolution feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activate: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial convolution followed by pointwise channel mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activate: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HybridResidualBlock(nn.Module):
    """Residual block with optional standard first convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        standard_first: bool = False,
    ) -> None:
        super().__init__()
        first_conv = StandardConv if standard_first else DepthwiseSeparableConv
        self.conv1 = first_conv(in_channels, out_channels, activate=True)
        self.conv2 = DepthwiseSeparableConv(
            out_channels, out_channels, activate=False
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        features = self.conv1(x)
        features = self.conv2(features)
        return self.activation(features + residual)


class SegmentationCNN(nn.Module):
    """Wider four-level hybrid residual U-Net.

    Compared with the 40,381-parameter hybrid model, this version increases
    channel capacity to (16, 24, 40, 56, 72), producing about 74K parameters.
    Standard convolutions remain in the first encoder and final decoder block;
    other convolutions are depthwise-separable to contain CPU cost.

    Input:  (B, 3, H, W), RGB float tensor in [0, 1].
    Output: (B, 1, H, W), raw logits.
    """

    CHANNELS = (16, 24, 40, 56, 72)

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4, cb = self.CHANNELS

        self.enc1 = HybridResidualBlock(3, c1, standard_first=True)
        self.enc2 = HybridResidualBlock(c1, c2)
        self.enc3 = HybridResidualBlock(c2, c3)
        self.enc4 = HybridResidualBlock(c3, c4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = HybridResidualBlock(c4, cb)

        self.dec4 = HybridResidualBlock(cb + c4, c4)
        self.dec3 = HybridResidualBlock(c4 + c3, c3)
        self.dec2 = HybridResidualBlock(c3 + c2, c2)
        self.dec1 = HybridResidualBlock(c2 + c1, c1, standard_first=True)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat((self.upsample(bottleneck, e4), e4), dim=1))
        d3 = self.dec3(torch.cat((self.upsample(d4, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self.upsample(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self.upsample(d2, e1), e1), dim=1))

        logits = self.head(d1)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SegmentationCNN().eval()
    print(f"Trainable parameters: {count_parameters(model):,}")
    with torch.inference_mode():
        for size in (128, 255, 256):
            output = model(torch.randn(1, 3, size, size))
            print(f"Input: {size}x{size} -> output: {tuple(output.shape)}")
