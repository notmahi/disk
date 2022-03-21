import numpy as np
import torch
import wandb
import utilities as utils


TANH_CONST = 4.0
LOG_FREQ = 10000
EPS_CONST = 1e-9


class UpdatingRewardModule(object):
    """
    Class for determining the reward of the incremental learning
    objective. This module simply keeps track of all the trajectory
    latents from all skills, and uses them to compute the reward
    for the current skill.
    """

    def __init__(
        self,
        transformed_obs_shape: int,
        max_episode_timesteps: int,
        saved_latent_per_skill: int,
        total_skills: int,
        max_running_obses: int = 10,
        slow_update_coeff: int = 10,
        device: str = "cuda",
        alpha: float = 1.0,
        beta: float = 1.0,
        use_t_obs: bool = False,
        use_t_vel: bool = True,
        topk=3,
    ):
        assert not (use_t_obs and use_t_vel), "Must use either t_obs or t_vel"
        assert use_t_obs or use_t_vel, "Must use either t_obs or t_vel"
        self.transformed_obs_dim = transformed_obs_shape[0]
        self.max_episode_timesteps = max_episode_timesteps
        self.saved_latent_per_skill = saved_latent_per_skill

        self.max_timesteps = 1

        self.slow_update_coeff = slow_update_coeff
        self.device = device
        self._init_alpha = alpha
        self._init_beta = beta
        self._circular_buffer_len = max_running_obses * max_episode_timesteps
        self._slow_update_len = self._circular_buffer_len // slow_update_coeff

        self._topk = topk

        # Set up the torch buffers.
        self._collected_trajectory_buffer = torch.zeros(
            *(saved_latent_per_skill * max_episode_timesteps, self.transformed_obs_dim),
            dtype=torch.float32,
            device=device,
        )
        self._circular_buffer = torch.zeros(
            *(self._circular_buffer_len, self.transformed_obs_dim),
            dtype=torch.float32,
            device=device,
        )
        self._circular_buffer_idx = 0
        self._collected_trajectory_buffer_idx = 0
        self._all_collected_trajectories = torch.zeros(
            *(0, self.transformed_obs_dim), device=device
        )
        self._reward_calls = 0.0
        self._saved_reward_call = (
            0.0  # self.max_episode_timesteps * self.saved_latent_per_skill
        )
        self._compactness_penalty_normalizer = 1.0
        self._diversity_reward_normalizer = 1.0
        self._consistency_penalties = []
        self._diversity_rewards = []

        self._obs_or_vel = (
            (lambda obs, next_obs: obs)
            if use_t_obs
            else (lambda obs, next_obs: next_obs - obs)
        )

    def register_logger(self, logger):
        self.logger = logger

    def current_len(self):
        pass

    @property
    def _current_circular_buffer_pos(self):
        return self._circular_buffer_idx % self._circular_buffer_len

    @property
    def _current_circular_buffer_len(self):
        return min(len(self._circular_buffer), self._circular_buffer_idx)

    def add_current(self, transformed_obs, obs, next_transformed_obs, *args):
        to_insert = torch.from_numpy(
            self._obs_or_vel(transformed_obs, next_transformed_obs)
        )
        self._circular_buffer[self._current_circular_buffer_pos] = to_insert
        self._circular_buffer_idx += 1

    def reset_collected_trajectories(self):
        """
        Empty out the collected trajectory buffer, since it may have stale data for previous skills.
        """
        self._all_collected_trajectories = torch.zeros(
            *(0, self.transformed_obs_dim), dtype=torch.float32, device=self.device
        )
        self._collected_trajectory_buffer_idx = 0

    def add_collected_trajectory(self, transformed_obs, next_transformed_obs):
        """
        Adds a single step of collected trajectory to the past-skill trajectory buffer.
        """
        to_insert = torch.from_numpy(
            self._obs_or_vel(transformed_obs, next_transformed_obs)
        )
        self._collected_trajectory_buffer[
            self._collected_trajectory_buffer_idx
        ] = to_insert
        self._collected_trajectory_buffer_idx += 1

    def save_collected_trajectories(self):
        """
        Saves the collected trajectories to the overall collected trajectory buffer.
        """
        self._all_collected_trajectories = torch.cat(
            (
                self._all_collected_trajectories,
                self._collected_trajectory_buffer[
                    : self._collected_trajectory_buffer_idx
                ],
            ),
            dim=0,
        )
        self._collected_trajectory_buffer *= 0.0
        self._collected_trajectory_buffer_idx = 0

    @property
    def temperature(self):
        if self._saved_reward_call == 0.0:
            return 1.0

        t = self._reward_calls / self._saved_reward_call
        return np.tanh((2 * t - 1) * TANH_CONST)

    def add_new_skill(self, num_steps_next_skill=None):
        if num_steps_next_skill is None:
            self._saved_reward_call = self._reward_calls
        else:
            self._saved_reward_call = num_steps_next_skill
        self._reward_calls = 0
        if len(self._consistency_penalties) == 0:
            self._consistency_penalties = [1.0]
        if len(self._diversity_rewards) == 0:
            self._diversity_rewards = [1.0]
        self._compactness_penalty_normalizer = 1.0 / (
            self._init_alpha * np.mean(self._consistency_penalties)
        )
        self._diversity_reward_normalizer = 1.0 / (
            self._init_beta * np.mean(self._diversity_rewards)
        )

    def get_rewards(
        self, next_t_obs_batch, t_obs_batch, batch_size=512, step=0, eval=False,
    ):
        query_batch = self._obs_or_vel(t_obs_batch, next_t_obs_batch)
        rewards = torch.zeros(
            *(len(query_batch), 1), dtype=torch.float32, device=self.device
        )
        if len(self._all_collected_trajectories):
            sampled_past_trajectories = self._sample_past_trajectories(batch_size)
            diversity_reward = _compute_entropy(
                query_batch, sampled_past_trajectories, topk=self._topk
            )
        else:
            diversity_reward = torch.ones_like(rewards)

        if self._current_circular_buffer_len > 0:
            sampled_current_trajectories = self._sample_current_trajectories(batch_size)
            consistency_penalty = _compute_entropy(
                query_batch, sampled_current_trajectories, topk=self._topk
            )
        else:
            consistency_penalty = torch.zeros_like(rewards)

        normalized_penalty = (
            -self._init_alpha
            * self.temperature
            * self._compactness_penalty_normalizer
            * consistency_penalty
        )
        normalized_bonus = (
            self._init_beta * self._diversity_reward_normalizer * diversity_reward
        )
        rewards = normalized_bonus + normalized_penalty
        self._log(
            penalty=normalized_penalty,
            bonus=normalized_bonus,
            reward=rewards,
            step=step,
        )
        return rewards

    def _sample_past_trajectories(self, num_samples: int) -> torch.Tensor:
        """
        Sample a batch of past-skill trajectories.
        """
        assert len(self._all_collected_trajectories) > 0
        indices = utils.perm_gpu_f32(
            pop_size=len(self._all_collected_trajectories), num_samples=num_samples
        )
        return self._all_collected_trajectories[indices]

    def _sample_current_trajectories(self, num_samples: int) -> torch.Tensor:
        """
        Sample a batch of current-skill trajectories.
        """
        assert self._circular_buffer_idx > 0
        sample_from = min(self._slow_update_len, self._current_circular_buffer_len)

        indices = utils.perm_gpu_f32(pop_size=sample_from, num_samples=num_samples)
        return self._circular_buffer[indices]

    def _log(self, penalty, bonus, reward, step=0):
        self._reward_calls += 1
        if step % LOG_FREQ == 0:
            mean_penalty = torch.mean(penalty).cpu().item()
            mean_bonus = torch.mean(bonus).cpu().item()
            mean_reward = torch.mean(reward).cpu().item()
            wandb.log(
                {
                    "update_step": step,
                    "reward": mean_reward,
                    "penalty": mean_penalty,
                    "bonus": mean_bonus,
                }
            )
            self._consistency_penalties.append(mean_penalty)
            self._diversity_rewards.append(mean_bonus)


# Helper functions
@torch.jit.script
def _compute_entropy(
    query_batch: torch.Tensor, buffer_sample_batch: torch.Tensor, topk: int = 3
) -> torch.Tensor:
    """
    Computes the entropy of the query batch with respect to the buffer sample batch.
    """
    query_batch = query_batch[:, None, :]
    buffer_sample_batch = buffer_sample_batch[None, :, :]
    l2_norm = torch.norm(
        (query_batch - buffer_sample_batch), p=2, dim=-1
    )  # (batch, buffer)
    topk_entropy, _ = torch.topk(l2_norm, topk, dim=-1, largest=False)
    entropy = topk_entropy[:, -1:]
    return entropy
