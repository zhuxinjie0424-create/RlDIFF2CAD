import torch
import torch.nn as nn


class ConditionFusion(nn.Module):
    """Fuse image, vec, and time embeddings into a single conditioning vector."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_img: torch.Tensor, z_vec: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        concatenated = torch.cat([z_img, z_vec, t_emb], dim=-1)
        return self.proj(concatenated)
