import torch
import torch.nn as nn


class RLCritic(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.embed(sequence)
        outputs, _ = self.encoder(x)
        values = self.value_head(outputs).squeeze(-1)
        return values
