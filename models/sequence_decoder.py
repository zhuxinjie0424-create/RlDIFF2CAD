import torch
import torch.nn as nn


class SequenceDecoder(nn.Module):
    def __init__(self, latent_dim: int, vocab_size: int, max_length: int = 64) -> None:
        super().__init__()
        self.max_length = max_length
        self.pos_embedding = nn.Embedding(max_length, latent_dim)
        self.decoder = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.out = nn.Linear(latent_dim, vocab_size)

    def forward(self, z_0: torch.Tensor) -> torch.Tensor:
        batch_size = z_0.size(0)
        positions = torch.arange(self.max_length, device=z_0.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0).repeat(batch_size, 1, 1)
        latent_seq = z_0.unsqueeze(1).repeat(1, self.max_length, 1) + pos_emb
        hidden, _ = self.decoder(latent_seq)
        logits = self.out(hidden)
        return logits
