"""Lightweight discrete diffusion scaffold for sketch token generation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence

from .tokens import SketchSequence, Token, TokenKind, Vocabulary


@dataclass
class DiffusionConfig:
    steps: int = 8
    noise_scale: float = 1.2
    vocab: Vocabulary = field(default_factory=Vocabulary)


class DiffusionModel:
    """Toy diffusion model that samples token sequences.

    The implementation keeps math intentionally simple: it operates on integer
    token indices, adds Gaussian-like noise, and performs greedy denoising. The
    class exposes a ``sample`` method that higher-level RL code can call during
    rollouts. Replace the sampler with a learned transformer-backed model to
    scale beyond this illustrative baseline.
    """

    def __init__(self, config: DiffusionConfig | None = None):
        self.config = config or DiffusionConfig()
        self.vocab = self.config.vocab
        self._token_bank = self.vocab.all_tokens()

    def _decode(self, idx: int) -> Token:
        idx = idx % len(self._token_bank)
        return self._token_bank[idx]

    def _noisy_latent(self, seq_len: int) -> List[float]:
        return [random.gauss(0.0, self.config.noise_scale) for _ in range(seq_len)]

    def _denoise_step(self, latent: List[float]) -> List[float]:
        decay = math.exp(-1 / max(self.config.steps, 1))
        return [value * decay + random.gauss(0.0, 0.05) for value in latent]

    def _latent_to_tokens(self, latent: List[float]) -> List[Token]:
        indices = [int(abs(value) * len(self._token_bank)) for value in latent]
        return [self._decode(idx) for idx in indices]

    def sample(self, seq_len: int = 12) -> SketchSequence:
        latent = self._noisy_latent(seq_len)
        for _ in range(self.config.steps):
            latent = self._denoise_step(latent)
        tokens = self._latent_to_tokens(latent)
        sketch = SketchSequence(vocabulary=self.vocab)
        sketch.extend(tokens)
        return sketch

    def score(self, sequence: SketchSequence) -> float:
        constraint_bonus = sum(1 for t in sequence if t.kind == TokenKind.CONSTRAINT)
        primitive_bonus = sum(1 for t in sequence if t.kind == TokenKind.PRIMITIVE)
        randomness = random.random() * 0.1
        return 0.1 * constraint_bonus + 0.05 * primitive_bonus + randomness

    def update_from_rewards(self, sequences: Sequence[SketchSequence], rewards: Sequence[float]) -> None:
        """Lightweight stand-in for gradient updates.

        We jitter the internal noise scale toward sequences that earned higher
        rewards to hint at a preference shift. Real implementations would back
        propagate through a diffusion loss combined with PPO gradients.
        """

        if not sequences:
            return
        avg_reward = sum(rewards) / len(rewards)
        self.config.noise_scale = max(0.1, min(2.5, self.config.noise_scale * (1 + 0.05 * (avg_reward - 0.5))))
