import hydra
import numpy as np
import torch
import torch.nn.functional as F
import utilities as utils
import wandb

from agent import Agent

LOG_FREQ = 100


class IncrementalAgent(Agent):
    """Incremental skill learning algorithm."""

    def __init__(
        self,
        obs_dim,
        t_obs_dim,
        action_dim,
        action_range,
        device,
        critic_cfg,
        actor_cfg,
        discount,
        init_temperature,
        alpha_lr,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        critic_target_update_frequency,
        batch_size,
        learnable_temperature,
    ):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.skill_actors = []

        self.critic_cfg = critic_cfg
        self.actor_cfg = actor_cfg
        self.init_temperature = init_temperature
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.actor_betas = actor_betas
        self.critic_lr = critic_lr
        self.critic_betas = critic_betas
        self.alpha_lr = alpha_lr
        self.alpha_betas = alpha_betas

        self.actor = None
        new_skill_actor = self.init_new_skill()
        self.current_skill_num = 0
        self.skill_actors.append(new_skill_actor)

        self.train()
        self.critic_target.train()

    def init_new_skill(self):
        self.critic = hydra.utils.instantiate(self.critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(self.critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Convert critic to torchscript
        self.critic = torch.jit.script(self.critic)
        self.critic_target = torch.jit.script(self.critic_target)

        if self.actor is not None:
            self.current_skill_num += 1

        self.actor = hydra.utils.instantiate(self.actor_cfg).to(self.device)
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -self.action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, betas=self.actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=self.critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=self.alpha_betas
        )

        return self.actor

    def register_reward_module(self, reward_module):
        self.reward_module = reward_module

    def get_skill(self, skill_index):
        assert skill_index <= self.current_skill_num, "Skill not learned yet"
        return self.skill_actors[skill_index]

    def add_new_skill(self, num_steps_next_skill=None):
        self.skill_actors[-1].eval()
        new_actor = self.init_new_skill()
        self.skill_actors.append(new_actor)
        self.train()
        self.critic_target.train()
        return new_actor

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False, skill_index=-1):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.unsqueeze(0)
        dist = self.skill_actors[skill_index](obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return utils.to_np(action[0])

    def update_critic(
        self, obs, action, reward, next_obs, not_done, logger, step,
    ):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        if step % LOG_FREQ == 0:
            logger.log("train_critic/loss", critic_loss, step)
            wandb.log(
                {"update_step": step, "train_critic/loss": critic_loss,}
            )

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if step % LOG_FREQ == 0:
            logger.log("train_actor/loss", actor_loss, step)
            logger.log("train_actor/entropy", -log_prob.mean(), step)
            wandb.log(
                {
                    "update_step": step,
                    "train_actor/loss": actor_loss,
                    "train_actor/target_entropy": self.target_entropy,
                    "train_actor/entropy": -log_prob.mean(),
                }
            )

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            if step % LOG_FREQ == 0:
                logger.log("train_alpha/loss", alpha_loss, step)
                logger.log("train_alpha/value", self.alpha, step)
                wandb.log(
                    {
                        "update_step": step,
                        "train_alpha/loss": alpha_loss,
                        "train_alpha/value": self.alpha,
                    }
                )

    def update(self, replay_buffer, logger, step):
        sample = replay_buffer.sample(self.batch_size)
        (
            obs,
            t_obs,
            action,
            reward,
            next_obs,
            next_t_obs,
            not_done,
            not_done_no_max,
        ) = sample

        reward = self.reward_module.get_rewards(next_t_obs, t_obs, step=step)

        if step % LOG_FREQ == 0:
            logger.log("train/batch_reward", reward.mean(), step)
            wandb.log(
                {"update_step": step, "reward": reward.mean(),}
            )

        self.update_critic(
            obs, action, reward, next_obs, not_done_no_max, logger, step,
        )

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

