import torch
import torch.nn as nn


class RLActor(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 256, max_length: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.policy = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.embed(sequence)
        outputs, _ = self.encoder(x)
        logits = self.policy(outputs)
        return logits
