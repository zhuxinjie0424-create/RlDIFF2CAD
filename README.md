# RlDIFF2CAD

RlDIFF2CAD is a reference implementation of a sequence-driven CAD sketch generator that combines a discrete diffusion model with PPO-style reinforcement learning. The project demonstrates how to represent sketches purely as token sequences, generate them with a diffusion prior, and refine the generator with structural rewards—without relying on intermediate geometric modalities or external constraint solvers.

## Key ideas
- **Unified token space:** primitives, constraints, and references share a single discrete vocabulary so the model never leaves sequence space.
- **Diffusion prior:** denoises latent token embeddings into structured sketch sequences that respect local syntax and layout statistics.
- **RL refinement:** PPO updates the diffusion model using rewards that measure structural validity, constraint consistency, connectivity, and compactness.
- **Closed-loop training:** diffusion → sequence → reward → PPO → diffusion forms an end-to-end optimization cycle.

## Repository layout
- `src/rldiff2cad/tokens.py`: token vocabulary, sequence containers, and utilities.
- `src/rldiff2cad/diffusion.py`: lightweight diffusion sampler scaffold operating on token embeddings.
- `src/rldiff2cad/reward.py`: reward functions for structure, constraint consistency, connectivity, and compactness.
- `src/rldiff2cad/rl.py`: PPO-style policy wrapper to fine-tune the diffusion model with rewards.
- `src/rldiff2cad/dataset.py`: loading and batching sequence datasets from JSONL files.
- `src/rldiff2cad/cli.py`: simple CLI for sampling sequences and running toy training loops.

## Quickstart
Install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Generate a few sample sequences with the diffusion prior:

```bash
python -m rldiff2cad.cli sample --num 3
```

Run a tiny toy training loop that alternates diffusion sampling and PPO updates:

```bash
python -m rldiff2cad.cli train --steps 5
```

> The implementation is intentionally lightweight and CPU-friendly, focusing on the algorithmic scaffold rather than model scale. You can swap in your own encoder/decoder or plug in a real transformer-based diffusion backbone where indicated in the code.

## License
MIT
