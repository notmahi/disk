import datetime
import os
import random

import dmc2gym
import gym
import numpy as np
import torch
import wandb
from dm_control import suite
from envs.square import SquareEnv
from envs.wrapper import DMCWrapper, _flatten_obs, _spec_to_box
from gym import spaces
from gym.envs.registration import register
from torch import nn

ENV_EXTRA_KWARGS = {
    "Ant-v3": {
        "xml_file": "/home/mahi/code/incremental_primitives/"
        "primitive_learning/config/env/ant.xml",
        "terminate_when_unhealthy": True,  # For hierarchical
    },
    "Ant-block": {
        "xml_file": "/home/mahi/code/incremental_primitives/"
        "primitive_learning/config/env/ant_block.xml",
        "terminate_when_unhealthy": False,  # For hierarchical
    },
    "Hopper-v3": {"terminate_when_unhealthy": False,},
    "Swimmer-v3": {
        "xml_file": "/home/mahi/code/incremental_primitives/"
        "primitive_learning/config/env/swimmer.xml",
    },
    "Humanoid-v3": {"terminate_when_unhealthy": False,},
}


# Subclass DMCWrapper for swimmer
class SwimmerWrapper(DMCWrapper):
    def __init__(
        self,
        n_links,
        task_kwargs=None,
        seed=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True,
    ):
        assert (
            "random" in task_kwargs
        ), "please specify a seed for deterministic behaviour"
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # create task
        self._env = suite.swimmer.swimmer(
            n_links=n_links, random=seed, environment_kwargs=environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        self._state_space = _spec_to_box(self._env.observation_spec().values())

        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs.get("random", 1))

    def reset(self):
        time_step = self._env.reset()
        # Make target a fixed point in space
        self._env._physics.named.model.geom_pos["target", "x"] = 10.0
        self._env._physics.named.model.geom_pos["target", "y"] = 10.0
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs


def make_env(cfg):
    """Helper function to create dm_control environment"""
    return make_env_from_env_name(cfg.env, cfg.seed)


def make_env_from_env_name(env_name, seed=0):
    """Helper function to create dm_control environment"""
    if env_name == "square":
        return SquareEnv(10, 10)

    group, env_name = env_name.split(".")
    if group == "dm":
        if not env_name.startswith("swimmer"):
            if env_name in ["ball_in_cup_catch", "point_mass_easy"]:
                domain_name = "_".join(env_name.split("_")[:-1])
                task_name = env_name.split("_")[-1]
            else:
                domain_name = env_name.split("_")[0]
                task_name = "_".join(env_name.split("_")[1:])

            env = dmc2gym.make(
                domain_name=domain_name,
                task_name=task_name,
                seed=seed,
                visualize_reward=True,
            )
            env.seed(seed)
        else:
            n_links = int(env_name.split("_")[1])
            env = _make_swimmer(n_links)
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1
    elif group == "gym":
        extra_kwargs = ENV_EXTRA_KWARGS.get(env_name, {})
        if env_name == "Ant-block":
            env = gym.make("Ant-v3", **extra_kwargs)
            # Set up all the blocks around the center.
            _dist = 10
            _num_blocks = 40
            for j in range(_num_blocks):
                angle = j * np.pi / (_num_blocks // 2.0)
                location = _dist * np.array([np.cos(angle), np.sin(angle), 0])
                env.model.body_pos[-j - 1] = location.copy()
        else:
            env = gym.make(env_name, **extra_kwargs)

    return env


def _make_swimmer(
    n_links=3,
    seed=1,
    visualize_reward=True,
    from_pixels=False,
    height=84,
    width=84,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
    channels_first=True,
):
    domain_name = "swimmer"
    task_name = str(n_links)
    env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seed)

    if from_pixels:
        assert (
            not visualize_reward
        ), "cannot use visualize reward when learning from pixels"

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if env_id not in gym.envs.registry.env_specs:
        task_kwargs = {}
        if seed is not None:
            task_kwargs["random"] = seed
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit
        register(
            id=env_id,
            entry_point="utils:SwimmerWrapper",
            kwargs=dict(
                n_links=n_links,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def get_current_timestamp():
    return str(datetime.datetime.now())[:19]


def compute_running_mean(previous_mean, value, count):
    new_mean = previous_mean * count
    new_mean += value
    new_mean /= count + 1.0
    return new_mean


def compute_rolling_mean(previous_mean, value, alpha=0.01):
    return (1 - alpha) * previous_mean + alpha * value


def scatter_batch(obs_batch, timestep_batch, max_timesteps):
    """
    Given a batch of observations and a batch of corresponding timesteps,
    create a sparse matrix where the correct indices hold the observations.
    """
    n_obs, n_dim = obs_batch.shape
    # Ignore timesteps entirely.
    scattered_batch = obs_batch.reshape(n_obs, 1, n_dim)
    return scattered_batch, None


def perm_gpu_f32(pop_size, num_samples):
    """Use torch.randperm to generate indices on a 32-bit GPU tensor."""
    return torch.randperm(pop_size, dtype=torch.int32, device="cuda")[
        :num_samples
    ].long()


def pointwise_entropy(points, k=10):
    x = torch.tensor(points)
    distance_matrix = torch.norm(x[:, None, :] - x[None, :, :], p=2, dim=-1)
    topk, _ = torch.topk(distance_matrix, k=min(k, len(x)), dim=1, largest=False,)
    topk_distance = torch.log(topk[:, -1:] + 1e-9)
    return topk_distance.mean().cpu().item()


def compute_cov(all_t_obs, num_skills):
    cov_matrix = np.cov(all_t_obs, rowvar=False)
    assert len(cov_matrix) == 2
    cov_det = np.linalg.det(cov_matrix)

    avg_det = []
    per_skill_evaluations = len(all_t_obs) // num_skills
    for idx in range(num_skills):
        start_idx = idx * per_skill_evaluations
        skill_cov = np.cov(
            all_t_obs[start_idx : start_idx + per_skill_evaluations], rowvar=False
        )
        skill_cov_det = np.linalg.det(skill_cov)
        avg_det.append(skill_cov_det)
    mean_cov_det = np.mean(avg_det)
    return cov_det, mean_cov_det
