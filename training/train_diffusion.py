from typing import Dict

import torch
from torch import optim
from torch.utils.data import DataLoader

from data.dataset import CADDataset
from models.diffusion_model import DiffusionModel
from models.fusion_module import ConditionFusion
from models.image_encoder import ImageEncoder
from models.sequence_decoder import SequenceDecoder
from models.vec_encoder import VecEncoder
from training.trainer_utils import diffusion_betas, sample_timesteps, sinusoidal_time_embedding


def diffusion_loss(noise_pred: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(noise_pred, noise)


def train_diffusion(
    dataset: CADDataset,
    config: Dict,
) -> Dict[str, torch.nn.Module]:
    device = torch.device(config["training"].get("device", "cpu"))
    latent_dim = config["model"]["latent_dim"]
    vocab_size = config["model"]["vocab_size"]
    diffusion_steps = config["model"]["diffusion_steps"]
    beta_start = config["model"]["beta_start"]
    beta_end = config["model"]["beta_end"]

    image_encoder = ImageEncoder(latent_dim, config["model"].get("image_channels", 3)).to(device)
    vec_encoder = VecEncoder(config["model"].get("vec_dim", 128), latent_dim).to(device)
    fusion = ConditionFusion(latent_dim).to(device)
    diffusion_model = DiffusionModel(latent_dim).to(device)
    decoder = SequenceDecoder(latent_dim, vocab_size).to(device)

    params = list(image_encoder.parameters()) + list(vec_encoder.parameters())
    params += list(fusion.parameters()) + list(diffusion_model.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=config["training"]["learning_rate"])

    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    betas = diffusion_betas(diffusion_steps, beta_start, beta_end).to(device)

    for epoch in range(config["training"]["num_epochs"]):
        for step, batch in enumerate(loader):
            image = batch["image"].to(device)
            vec = batch["vec"].to(device)
            sequence = batch["sequence"].to(device)

            optimizer.zero_grad()

            z_img = image_encoder(image)
            z_vec = vec_encoder(vec)

            timesteps = sample_timesteps(image.size(0), diffusion_steps, device)
            t_emb = sinusoidal_time_embedding(timesteps, latent_dim)

            noise = torch.randn_like(z_img)
            alpha_t = (1.0 - betas[timesteps]).sqrt()
            alpha_bar = alpha_t.cumprod(dim=0)
            z_t = alpha_bar.unsqueeze(-1) * z_img + (1 - alpha_bar).unsqueeze(-1) * noise

            cond = fusion(z_img, z_vec, t_emb)
            noise_pred = diffusion_model(z_t, cond, t_emb)

            loss = diffusion_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

            if step % config["training"]["log_interval"] == 0:
                print(f"Epoch {epoch} Step {step} Diffusion Loss: {loss.item():.4f}")

    return {
        "image_encoder": image_encoder,
        "vec_encoder": vec_encoder,
        "fusion": fusion,
        "diffusion_model": diffusion_model,
        "decoder": decoder,
    }
