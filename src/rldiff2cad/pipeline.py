from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from torch import nn

from .config import DiffusionConfig


@dataclass
class VectorConditioner(nn.Module):
    """Lightweight projector to align vector embeddings with text embeddings."""

    input_dim: int
    hidden_dim: Optional[int] = None

    def __post_init__(self) -> None:
        super().__init__()
        hidden_dim = self.hidden_dim or self.input_dim
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, vector: torch.Tensor, target_dim: int) -> torch.Tensor:
        if vector.ndim != 2:
            raise ValueError("VectorConditioner expects input shape (batch, features).")
        projected = self.projection(vector)
        if projected.shape[1] != target_dim:
            projected = nn.functional.interpolate(
                projected.unsqueeze(1), size=target_dim, mode="linear", align_corners=False
            ).squeeze(1)
        return projected


class ConditionedDiffusionPipeline:
    """Stable Diffusion img2img pipeline conditioned on vector embeddings."""

    def __init__(self, config: DiffusionConfig, vector_dim: int) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.model_id, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        self.vector_conditioner = VectorConditioner(input_dim=vector_dim)

    def _encode_prompt_with_vector(self, prompt: str, vector: np.ndarray) -> torch.Tensor:
        prompt_embeds = self.pipeline._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        vector_tensor = torch.from_numpy(vector).to(self.device).unsqueeze(0)
        cond = self.vector_conditioner(vector_tensor, prompt_embeds.shape[-1])
        return prompt_embeds + cond.unsqueeze(1)

    def __call__(
        self, prompt: str, init_image: Image.Image, vector: np.ndarray, output_path: Optional[str] = None
    ) -> Image.Image:
        if self.config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        else:
            generator = None

        prompt_embeds = self._encode_prompt_with_vector(prompt, vector)
        negative_embeds = self.pipeline._encode_prompt(
            prompt="", device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True
        )

        result = self.pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            image=init_image,
            strength=self.config.strength,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            generator=generator,
        )

        image = result.images[0]
        if output_path:
            image.save(output_path)
        return image
