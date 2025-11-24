import torch
import torch.nn as nn


class VecEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.net(vec)
