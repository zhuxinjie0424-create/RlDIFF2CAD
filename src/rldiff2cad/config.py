from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _resolve_path(value: str | Path | None) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


@dataclass
class DiffusionConfig:
    """Configuration for running the conditioned diffusion pipeline."""

    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    strength: float = 0.85
    image_condition_weight: float = 0.5
    seed: Optional[int] = None
    vector_path: Optional[Path] = None
    image_path: Optional[Path] = None

    def resolved_vector_path(self) -> Optional[Path]:
        return _resolve_path(self.vector_path)

    def resolved_image_path(self) -> Optional[Path]:
        return _resolve_path(self.image_path)
