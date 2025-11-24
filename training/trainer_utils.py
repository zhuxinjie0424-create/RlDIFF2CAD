import math
from typing import Tuple

import torch


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def sample_timesteps(batch_size: int, total_steps: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, total_steps, (batch_size,), device=device)


def diffusion_betas(num_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def linear_noise_schedule(timesteps: torch.Tensor, betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    beta_t = betas[timesteps]
    alpha_t = 1.0 - beta_t
    alpha_bar = torch.cumprod(alpha_t, dim=0)
    return alpha_t, alpha_bar
