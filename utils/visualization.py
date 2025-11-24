from typing import List

import matplotlib.pyplot as plt
import torch


def plot_sequence(logits: torch.Tensor, token_list: List[str], path: str) -> None:
    probs = logits.softmax(dim=-1).mean(dim=0).cpu().detach().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(probs.T, aspect="auto", cmap="viridis")
    plt.yticks(range(len(token_list)), token_list)
    plt.xlabel("Time")
    plt.ylabel("Token")
    plt.colorbar(label="Probability")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
