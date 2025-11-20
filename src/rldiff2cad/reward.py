"""Reward shaping utilities for CAD sketch sequences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .tokens import SketchSequence, TokenKind


@dataclass
class RewardConfig:
    structure_validity: float = 1.0
    constraint_consistency: float = 1.0
    connectivity: float = 0.5
    compactness: float = 0.2


def _structure_validity(seq: SketchSequence) -> float:
    if not seq.tokens:
        return 0.0
    primitive_count = sum(1 for t in seq if t.kind == TokenKind.PRIMITIVE)
    constraint_count = sum(1 for t in seq if t.kind == TokenKind.CONSTRAINT)
    has_refs = any(t.kind == TokenKind.REFERENCE for t in seq)
    return min(1.0, 0.4 + 0.1 * primitive_count + 0.05 * constraint_count + 0.1 * has_refs)


def _constraint_consistency(seq: SketchSequence) -> float:
    constraint_count = sum(1 for t in seq if t.kind == TokenKind.CONSTRAINT)
    primitive_count = sum(1 for t in seq if t.kind == TokenKind.PRIMITIVE)
    if constraint_count == 0:
        return 0.1
    ratio = constraint_count / max(primitive_count, 1)
    return max(0.0, 1.0 - abs(ratio - 0.8))


def _connectivity(seq: SketchSequence) -> float:
    ref_count = sum(1 for t in seq if t.kind == TokenKind.REFERENCE)
    primitive_count = sum(1 for t in seq if t.kind == TokenKind.PRIMITIVE)
    return min(1.0, 0.2 + 0.1 * ref_count + 0.1 * primitive_count)


def _compactness(seq: SketchSequence) -> float:
    return 1.0 / (1.0 + 0.05 * len(seq.tokens))


def compute_rewards(seq: SketchSequence, config: RewardConfig | None = None) -> Dict[str, float]:
    cfg = config or RewardConfig()
    rewards = {
        "structure_validity": _structure_validity(seq),
        "constraint_consistency": _constraint_consistency(seq),
        "connectivity": _connectivity(seq),
        "compactness": _compactness(seq),
    }
    total = (
        cfg.structure_validity * rewards["structure_validity"]
        + cfg.constraint_consistency * rewards["constraint_consistency"]
        + cfg.connectivity * rewards["connectivity"]
        + cfg.compactness * rewards["compactness"]
    )
    rewards["total"] = total
    return rewards
