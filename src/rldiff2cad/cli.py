"""Entry points for sampling and toy training."""
from __future__ import annotations

import argparse
from typing import Iterable

from .diffusion import DiffusionConfig, DiffusionModel
from .reward import compute_rewards
from .rl import PPOConfig, PPOTrainer


def _print_sequences(seqs: Iterable):
    for i, seq in enumerate(seqs, 1):
        rewards = compute_rewards(seq)
        print(f"[{i}] {seq.summary()} reward={rewards['total']:.3f}")
        print("  " + ", ".join(seq.to_strings()))


def run_sample(args: argparse.Namespace) -> None:
    diffusion = DiffusionModel(DiffusionConfig())
    samples = [diffusion.sample(seq_len=args.seq_len) for _ in range(args.num)]
    _print_sequences(samples)


def run_train(args: argparse.Namespace) -> None:
    diffusion = DiffusionModel(DiffusionConfig())
    trainer = PPOTrainer(diffusion, PPOConfig())
    rewards = trainer.train(steps=args.steps)
    print("Training complete. Reward trace:")
    print([round(r, 3) for r in rewards])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RlDIFF2CAD utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sample_p = sub.add_parser("sample", help="Generate samples from the diffusion prior")
    sample_p.add_argument("--num", type=int, default=3)
    sample_p.add_argument("--seq-len", type=int, default=12)
    sample_p.set_defaults(func=run_sample)

    train_p = sub.add_parser("train", help="Run a toy PPO training loop")
    train_p.add_argument("--steps", type=int, default=5)
    train_p.set_defaults(func=run_train)

    return parser


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI glue
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
