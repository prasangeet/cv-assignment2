"""Hybrid residual U-Net for efficient binary salient-object segmentation."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class StandardConv(nn.Module):
    """Standard 3x3 convolution, used where high-resolution feature mixing matters."""

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
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial filtering followed by pointwise channel mixing."""

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
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HybridResidualBlock(nn.Module):
    """Residual block with configurable standard/depthwise first convolution.

    A standard first convolution can be enabled in the highest-resolution
    encoder/decoder blocks. All remaining convolutions are depthwise-separable.
    """

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
            out_channels,
            out_channels,
            activate=False,
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
    """Four-level hybrid residual U-Net.

    Input:  (B, 3, H, W), RGB float tensor in [0, 1].
    Output: (B, 1, H, W), raw logits.

    Standard convolution is retained in the first encoder and final decoder
    refinement blocks to preserve high-resolution spatial detail. Deeper
    encoder/decoder blocks use depthwise-separable convolutions to reduce
    parameters and compute. Direct skip connections and residual projections
    are retained throughout.
    """

    def __init__(self) -> None:
        super().__init__()

        # Four encoder levels. The first block keeps a standard 3x3 feature
        # extractor because it operates directly on RGB/high-resolution data.
        self.enc1 = HybridResidualBlock(3, 12, standard_first=True)
        self.enc2 = HybridResidualBlock(12, 20)
        self.enc3 = HybridResidualBlock(20, 28)
        self.enc4 = HybridResidualBlock(28, 40)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = HybridResidualBlock(40, 48)

        # Decoder. Keep a standard first convolution only at the final,
        # highest-resolution refinement stage for boundary/detail recovery.
        self.dec4 = HybridResidualBlock(48 + 40, 40)
        self.dec3 = HybridResidualBlock(40 + 28, 28)
        self.dec2 = HybridResidualBlock(28 + 20, 20)
        self.dec1 = HybridResidualBlock(20 + 12, 12, standard_first=True)

        self.head = nn.Conv2d(12, 1, kernel_size=1)

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
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = SegmentationCNN().eval()
    print(f"Trainable parameters: {count_parameters(model):,}")
    with torch.inference_mode():
        for size in (128, 255, 256):
            output = model(torch.randn(1, 3, size, size))
            print(f"Input: {size}x{size} -> output: {tuple(output.shape)}")
