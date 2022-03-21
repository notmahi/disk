import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from typing import Dict

import utilities as utils


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade
        # the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable,
        # see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds,
    ):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        o_dim = obs_dim
        self.trunk = utils.mlp(o_dim, hidden_dim, 2 * action_dim, hidden_depth)
        self._action_dim = action_dim
        self.outputs: Dict[str : torch.Tensor] = dict()
        self.apply(utils.weight_init)

        self._broken_leg_multiplier = None

    def forward(self, obs):
        mu_log_std = self.trunk(obs)
        mu, log_std = (
            mu_log_std[..., : self._action_dim],
            mu_log_std[..., self._action_dim :],
        )

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist
