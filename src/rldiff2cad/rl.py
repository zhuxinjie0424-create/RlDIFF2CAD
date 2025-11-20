"""PPO-style trainer that refines the diffusion model with structural rewards."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

from .diffusion import DiffusionModel
from .reward import RewardConfig, compute_rewards
from .tokens import SketchSequence


@dataclass
class PPOConfig:
    steps_per_update: int = 4
    reward_config: RewardConfig = field(default_factory=RewardConfig)


class PPOTrainer:
    """Lightweight PPO loop that treats the diffusion model as a policy."""

    def __init__(self, diffusion: DiffusionModel, config: PPOConfig | None = None):
        self.diffusion = diffusion
        self.config = config or PPOConfig()

    def rollout(self) -> Tuple[SketchSequence, float]:
        seq = self.diffusion.sample()
        rewards = compute_rewards(seq, self.config.reward_config)
        return seq, rewards["total"]

    def update(self, sequences: Iterable[SketchSequence], rewards: Iterable[float]) -> None:
        seqs = list(sequences)
        rewards = list(rewards)
        if not seqs:
            return
        self.diffusion.update_from_rewards(seqs, rewards)

    def train(self, steps: int = 10) -> List[float]:
        history: List[float] = []
        for _ in range(steps):
            batch_seqs: List[SketchSequence] = []
            batch_rewards: List[float] = []
            for _ in range(self.config.steps_per_update):
                seq, reward = self.rollout()
                batch_seqs.append(seq)
                batch_rewards.append(reward)
                history.append(reward)
            self.update(batch_seqs, batch_rewards)
        return history
