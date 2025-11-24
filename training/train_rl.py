from typing import Dict

import torch
from torch import optim
from torch.distributions import Categorical

from models.rl_actor import RLActor
from models.rl_critic import RLCritic
from models.reward_fn import compute_reward


def _compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t] if t + 1 < values.size(1) else rewards[:, t] - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    return advantages


def train_rl(models: Dict[str, torch.nn.Module], sequences: torch.Tensor, config: Dict) -> Dict[str, torch.nn.Module]:
    device = next(models["image_encoder"].parameters()).device
    vocab_size = config["model"]["vocab_size"]
    actor = RLActor(vocab_size).to(device)
    critic = RLCritic(vocab_size).to(device)

    actor_optim = optim.Adam(actor.parameters(), lr=config["training"]["learning_rate"])
    critic_optim = optim.Adam(critic.parameters(), lr=config["training"]["learning_rate"])

    clip_range = config["rl"]["clip_range"]
    gamma = config["rl"]["gamma"]
    lam = config["rl"]["gae_lambda"]
    entropy_coef = config["rl"]["entropy_coef"]
    value_coef = config["rl"]["value_coef"]

    for _ in range(config["rl"]["ppo_epochs"]):
        logits = actor(sequences)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        values = critic(sequences)
        rewards = compute_reward(actions)

        advantages = _compute_advantages(rewards.unsqueeze(1), values.unsqueeze(1), gamma, lam)
        returns = advantages + values.unsqueeze(1)

        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        actor_loss = -(torch.min(surr1, surr2) + entropy_coef * dist.entropy()).mean()

        critic_loss = value_coef * (returns.squeeze(1) - values).pow(2).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config["rl"]["max_grad_norm"])
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config["rl"]["max_grad_norm"])
        critic_optim.step()

    models.update({"actor": actor, "critic": critic})
    return models
