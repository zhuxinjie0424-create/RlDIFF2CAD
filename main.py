import json
from pathlib import Path

import torch
import yaml

from data.dataset import CADDataset
from data.preprocess import ensure_data_dir
from data.tokenizer import Tokenizer
from models.sequence_decoder import SequenceDecoder
from training.train_diffusion import train_diffusion
from training.train_rl import train_rl


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def diffusion_generate(models, dataset, config):
    device = torch.device(config["training"].get("device", "cpu"))
    betas = torch.linspace(
        config["model"]["beta_start"],
        config["model"]["beta_end"],
        config["model"]["diffusion_steps"],
        device=device,
    )
    tokenizer = dataset.tokenizer
    decoder: SequenceDecoder = models["decoder"]
    decoder.eval()
    sequences = []
    with torch.no_grad():
        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(device)
            vec = sample["vec"].unsqueeze(0).to(device)
            z_img = models["image_encoder"](image)
            z_vec = models["vec_encoder"](vec)
            z_t = torch.randn_like(z_img)
            for t in reversed(range(config["model"]["diffusion_steps"])):
                t_tensor = torch.tensor([t], device=device)
                t_emb = torch.nn.functional.one_hot(t_tensor, num_classes=config["model"]["latent_dim"]).float()
                cond = models["fusion"](z_img, z_vec, t_emb)
                z_t, _ = models["diffusion_model"].predict_prev(z_t, cond, t_emb, betas[t])
            logits = decoder(z_t)
            sequence = torch.argmax(logits, dim=-1).squeeze(0)
            sequences.append(sequence.cpu())
    return torch.stack(sequences, dim=0), tokenizer


def main():
    config = load_config("config/config.yaml")
    ensure_data_dir("data")

    # placeholder empty dataset
    samples = tuple()
    dataset = CADDataset(samples)

    models = train_diffusion(dataset, config)

    sequences, tokenizer = diffusion_generate(models, dataset, config)
    models = train_rl(models, sequences, config)

    Path("artifacts").mkdir(exist_ok=True)
    tokenizer.save("artifacts/tokenizer.json")
    torch.save({k: v.state_dict() for k, v in models.items()}, "artifacts/model.pt")
    with open("artifacts/config_dump.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
