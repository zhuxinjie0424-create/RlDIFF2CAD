from typing import Tuple

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class DiffusionModel(nn.Module):
    """Simple residual diffusion core that predicts noise."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(latent_dim * 2, latent_dim)
        self.resblocks = nn.Sequential(
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim),
        )
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_t: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([z_t, cond], dim=-1)
        hidden = self.input_proj(merged)
        hidden = hidden + t_emb  # inject timestep
        hidden = self.resblocks(hidden)
        return self.output_proj(hidden)

    def predict_prev(self, z_t: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor, beta_t: float) -> Tuple[torch.Tensor, torch.Tensor]:
        noise_pred = self.forward(z_t, cond, t_emb)
        z_prev = z_t - beta_t * noise_pred
        return z_prev, noise_pred
