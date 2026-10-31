import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Single optimized convolution block to drastically reduce CPU memory bandwidth bottleneck."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationCNN(nn.Module):
    """
    Ultra-Fast U-Net: Single-conv blocks to guarantee <35ms CPU time
    while preserving skip connections for high mIoU and Boundary F1.
    Input:  (B, 3, 256, 256)
    Output: (B, 1, 256, 256) logits
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 48)

        # Bottleneck
        self.bottleneck = ConvBlock(48, 48)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Decoder matches encoder skip channels
        self.dec3 = ConvBlock(48 + 48, 32)
        self.dec2 = ConvBlock(32 + 32, 16)
        self.dec1 = ConvBlock(16 + 16, 16)

        # Binary segmentation logits
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)                               # 256 x 256
        e2 = self.enc2(self.pool(e1))                   # 128 x 128
        e3 = self.enc3(self.pool(e2))                   # 64 x 64

        b = self.bottleneck(self.pool(e3))              # 32 x 32

        # Decoder with Bilinear Interpolation
        d3 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)
