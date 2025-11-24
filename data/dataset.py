from typing import Any, Dict, Tuple

import torch
from torch.utils.data import Dataset

from data.preprocess import load_image, load_vec
from data.tokenizer import Tokenizer


def _dummy_sequence(tokenizer: Tokenizer, length: int = 32) -> torch.Tensor:
    ids = [i % len(tokenizer.tokens) for i in range(length)]
    return torch.tensor(ids, dtype=torch.long)


class CADDataset(Dataset):
    """Tiny dataset placeholder for CAD diffusion + RL."""

    def __init__(self, samples: Tuple[Tuple[str, str], ...]) -> None:
        self.samples = samples
        self.tokenizer = Tokenizer()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, vec_path = self.samples[idx]
        image = load_image(image_path)
        vec = load_vec(vec_path)
        sequence = _dummy_sequence(self.tokenizer)
        return {"image": image, "vec": vec, "sequence": sequence}
