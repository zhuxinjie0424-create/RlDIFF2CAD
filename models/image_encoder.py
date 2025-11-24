import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Lightweight image encoder emulating Stable Diffusion's encoder interface."""

    def __init__(self, latent_dim: int, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, latent_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.net(image)
        flattened = feats.flatten(1)
        return self.proj(flattened)
