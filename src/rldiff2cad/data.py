from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_condition_inputs(image_path: Path, vector_path: Path) -> Tuple[Image.Image, np.ndarray]:
    """Load a conditioning image and vector embedding.

    Parameters
    ----------
    image_path:
        Path to a CAD render or sketch used for img2img conditioning.
    vector_path:
        Path to a serialized vector embedding (.npy).
    """

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    image = Image.open(image_path).convert("RGB")

    vector = np.load(vector_path)
    if vector.ndim != 1:
        raise ValueError("Vector embedding must be one-dimensional.")

    return image, vector.astype(np.float32)
