import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, hidden_dim=192, latent_dim=144, num_actions=7):
        super().__init__()
        feat_dim = hidden_dim + latent_dim
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ELU(),
            nn.Linear(feat_dim, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ELU(),
            nn.Linear(feat_dim, 1),
        )

    def action_logits(self, h, z):
        return self.actor(torch.cat([h, z], dim=-1))

    def value(self, h, z):
        return self.critic(torch.cat([h, z], dim=-1)).squeeze(-1)

    def act(self, h, z):
        dist = Categorical(logits=self.action_logits(h, z))
        action = dist.sample()
        return action, dist.log_prob(action)
