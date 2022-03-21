import torch
from torch import nn

import utilities as utils


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        o_dim = obs_dim
        self.Q1 = utils.mlp(o_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(o_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        to_cat = [obs, action]
        obs_action = torch.cat(to_cat, dim=-1)

        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2
