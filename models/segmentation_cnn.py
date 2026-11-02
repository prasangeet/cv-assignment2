"""Residual multi-scale U-Net with deep supervision for 128x128 saliency segmentation."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Two 3x3 convolutions with a projected residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.shortcut(x))


class DilatedContext(nn.Module):
    """Low-cost multi-scale context aggregation at the bottleneck."""

    def __init__(self, channels: int, branch_channels: int = 16) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in (1, 2, 4):
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        branch_channels,
                        branch_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=branch_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.activation(x + self.project(multi_scale))


class SegmentationCNN(nn.Module):
    """Three-level residual U-Net with multi-scale context and deep supervision.

    Input: (B, 3, H, W), RGB float tensor in [0, 1].
    Output: (B, 1, H, W), raw logits by default.

    Call ``model(x, return_aux=True)`` during training to receive
    ``(main_logits, aux_logits_1_4, aux_logits_1_2)``. The ordinary inference
    interface remains a single logits tensor for compatibility with evaluate.py.
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder: modestly redistribute capacity toward deeper features.
        self.enc1 = ResidualConvBlock(3, 16)
        self.enc2 = ResidualConvBlock(16, 24)
        self.enc3 = ResidualConvBlock(24, 40)
        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = ResidualConvBlock(40, 56)
        self.context = DilatedContext(56, branch_channels=16)

        self.dec3 = ResidualConvBlock(56 + 40, 40)
        self.dec2 = ResidualConvBlock(40 + 24, 24)
        self.dec1 = ResidualConvBlock(24 + 16, 16)

        self.head = nn.Conv2d(16, 1, kernel_size=1)
        self.aux_quarter = nn.Conv2d(40, 1, kernel_size=1)
        self.aux_half = nn.Conv2d(24, 1, kernel_size=1)

    @staticmethod
    def upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        bottleneck = self.context(self.bottleneck(self.pool(e3)))

        d3 = self.dec3(torch.cat((self.upsample(bottleneck, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self.upsample(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self.upsample(d2, e1), e1), dim=1))

        main_logits = self.head(d1)
        if main_logits.shape[-2:] != original_size:
            main_logits = F.interpolate(
                main_logits, size=original_size, mode="bilinear", align_corners=False
            )
        if not return_aux:
            return main_logits

        aux_quarter = F.interpolate(
            self.aux_quarter(d3), size=original_size, mode="bilinear", align_corners=False
        )
        aux_half = F.interpolate(
            self.aux_half(d2), size=original_size, mode="bilinear", align_corners=False
        )
        return main_logits, aux_quarter, aux_half


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SegmentationCNN().eval()
    print(f"Trainable parameters: {count_parameters(model):,}")
    with torch.inference_mode():
        for side in (128, 127, 255):
            x = torch.randn(1, 3, side, side)
            y = model(x)
            aux = model(x, return_aux=True)
            print(f"{side}x{side}: output={tuple(y.shape)}, aux={[tuple(t.shape) for t in aux]}")
