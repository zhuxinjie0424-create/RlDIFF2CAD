import torch


def constraint_consistency(sequence: torch.Tensor) -> torch.Tensor:
    return sequence.float().mean(dim=-1)


def connectivity_score(sequence: torch.Tensor) -> torch.Tensor:
    return (sequence % 2).float().mean(dim=-1)


def grammar_score(sequence: torch.Tensor) -> torch.Tensor:
    return (sequence < sequence.max(dim=-1, keepdim=True).values).float().mean(dim=-1)


def compactness_score(sequence: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + sequence.float().std(dim=-1))


def compute_reward(sequence: torch.Tensor, weights=None) -> torch.Tensor:
    if weights is None:
        weights = (1.0, 1.0, 1.0, 1.0)
    w1, w2, w3, w4 = weights
    reward = (
        w1 * constraint_consistency(sequence)
        + w2 * connectivity_score(sequence)
        + w3 * grammar_score(sequence)
        + w4 * compactness_score(sequence)
    )
    return reward
