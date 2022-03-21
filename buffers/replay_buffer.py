import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, t_obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.t_obses = np.empty((capacity, *t_obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_t_obses = np.empty((capacity, *t_obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(
        self, obs, t_obs, action, reward, next_obs, next_t_obs, done, done_no_max,
    ):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.t_obses[self.idx], t_obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_t_obses[self.idx], next_t_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def purge_frac(self, frac=0.5):
        to_keep = int((1.0 - frac) * self.__len__())
        idxs = np.random.randint(0, self.__len__(), size=to_keep)

        self.obses[:to_keep] = self.obses[idxs]
        self.t_obses[:to_keep] = self.t_obses[idxs]
        self.actions[:to_keep] = self.actions[idxs]
        self.rewards[:to_keep] = self.rewards[idxs]
        self.next_obses[:to_keep] = self.next_obses[idxs]
        self.next_t_obses[:to_keep] = self.next_t_obses[idxs]
        self.not_dones[:to_keep] = self.not_dones[idxs]
        self.not_dones_no_max[:to_keep] = self.not_dones_no_max[idxs]

        self.idx = to_keep
        self.full = False

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.__len__(), size=batch_size)

        obses = torch.from_numpy(self.obses[idxs]).to(self.device)
        t_obses = torch.from_numpy(self.t_obses[idxs]).to(self.device)
        actions = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)
        next_obses = torch.from_numpy(self.next_obses[idxs]).to(self.device)
        next_t_obses = torch.from_numpy(self.next_t_obses[idxs]).to(self.device)
        not_dones = torch.from_numpy(self.not_dones[idxs]).to(self.device)
        not_dones_no_max = torch.from_numpy(self.not_dones_no_max[idxs]).to(self.device)

        return (
            obses,
            t_obses,
            actions,
            rewards,
            next_obses,
            next_t_obses,
            not_dones,
            not_dones_no_max,
        )

