from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def load_image(path: str, image_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Load and normalize an image for the image encoder."""
    img = Image.open(path).convert("RGB").resize(image_size)
    arr = np.asarray(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def load_vec(path: str) -> torch.Tensor:
    """Placeholder for loading CAD vector/h5 data."""
    data = np.load(path)
    return torch.from_numpy(data.astype("float32"))


def preprocess_sample(image_path: str, vec_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_image(image_path), load_vec(vec_path)


def ensure_data_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
