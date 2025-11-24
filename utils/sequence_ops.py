from typing import Tuple

import torch


def apply_action(sequence: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(sequence, dtype=torch.bool)
    mask[:, : action.size(1)] = True
    updated = torch.where(mask, action, sequence)
    return updated


def sequence_to_representation(sequence: torch.Tensor) -> torch.Tensor:
    return sequence.float()


def enforce_constraints(sequence: torch.Tensor, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    clipped = sequence.clamp(0, vocab_size - 1)
    mask = (clipped == sequence).float()
    return clipped, mask
