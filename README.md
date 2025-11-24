# RlDIFF2CAD

This repository implements a modular scaffold for a CAD-oriented diffusion and reinforcement learning system. The layout mirrors the design described in the prompt, including encoders, diffusion core, sequence decoder, PPO-style RL components, and helper utilities.

## Structure
- `config/`: YAML configuration for model, training, and RL hyperparameters.
- `data/`: Lightweight preprocessing helpers, tokenizer, and a placeholder dataset.
- `models/`: Encoders, condition fusion, diffusion core, sequence decoder, RL actor/critic, and reward computation.
- `training/`: Training utilities and stubbed loops for diffusion pretraining and RL fine-tuning.
- `utils/`: Logging, visualization, and sequence operations.
- `main.py`: Entry point wiring diffusion generation and RL refinement, persisting artifacts to `artifacts/`.

## Running
The code uses PyTorch. Install dependencies (e.g., `torch`, `pyyaml`, `numpy`, `pillow`, `matplotlib`) then run:

```bash
python main.py
```

The current data pipeline is a placeholder; populate `data/` with image/vector samples and adjust `config/config.yaml` to suit your environment.
