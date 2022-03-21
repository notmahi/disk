#!/usr/bin/env python3
import os
import re
import time

import hydra
import omegaconf
import torch
from tqdm import tqdm

import wandb
from envs.obs_transforms import (
    OBS_TRANSFORMS,
    RECORD_TRANSFORMS,
    obs_transform_default,
    record_transform_default,
)
import buffers
import utilities as utils


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        wandb.init(
            project="disk",
            group=cfg.group,
            tags=cfg.tags,
            monitor_gym=True,
            config=omegaconf.OmegaConf.to_container(self.cfg),
        )

        utils.set_seed_everywhere(cfg.seed)
        self.logger = utils.Logger(
            self.work_dir, save_tb=cfg.log_save_tb, log_frequency=cfg.log_frequency
        )
        self.device = torch.device(cfg.device)
        self.env = utils.make_env(cfg)

        _, self.env_name = cfg.env.split(".")

        self._setup_blocks_if_needed()
        self._setup_number_of_skills()
        self._setup_transformed_obses()
        self.agent = hydra.utils.instantiate(self.cfg.agent)

        self._setup_buffers()
        self._setup_video_recording()

        self.step = 0
        self.episode = 0

    def evaluate(self, record=True):
        average_episode_reward = 0
        self.env._max_episode_steps = self.cfg.max_test_episode_steps

        self.video_recorder.init(enabled=record)
        for skill_idx in tqdm(range(self.agent.current_skill_num + 1)):
            self.video_recorder.init_new_skill()
            self.video_recorder.current_skill = skill_idx
            for episode in range(self.cfg.num_eval_episodes):
                self.video_recorder.record_blank(f"Ep {episode} skill {skill_idx}")
                obs = self.env.reset()
                if self.env.viewer:
                    self.env.viewer.cam.distance = 10.0
                self.agent.reset()
                done = False
                episode_reward = 0
                while not done:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(
                            obs, sample=False, skill_index=skill_idx,
                        )
                    next_obs, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    obs = next_obs.copy()
                    self.video_recorder.record(self.env, no_mujoco=True, make_fig=True)

                average_episode_reward += episode_reward

        self.video_recorder.save(f"Step_{self.step}.mp4")
        average_episode_reward /= self.cfg.num_eval_episodes * (
            self.agent.current_skill_num + 1
        )
        self.logger.log("eval/episode_reward", average_episode_reward, self.step)
        wandb.log({"rewards/eval_reward": average_episode_reward})
        self.logger.dump(self.step)
        self.env._max_episode_steps = self.cfg.max_episode_steps

    def collect_trajectories(self, skill_idx=-1):
        # Collect trajectories from a completed skill
        # and save it to the skill replay buffer.
        average_episode_reward = 0
        self.env._max_episode_steps = self.cfg.max_episode_steps
        for _ in range(self.collected_trajectories):
            obs = self.env.reset()
            transformed_obs = self.transform_obs(obs)
            self.agent.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False, skill_index=skill_idx,)
                transformed_obs = self.transform_obs(obs)
                next_obs, reward, done, _ = self.env.step(action)
                next_transformed_obs = self.transform_obs(next_obs)
                episode_reward += reward
                episode_step += 1
                self.reward_module.add_collected_trajectory(
                    transformed_obs, next_transformed_obs
                )
                obs = next_obs.copy()
                transformed_obs = next_transformed_obs.copy()
            average_episode_reward += episode_reward

    def run(self):
        episode, episode_reward, done = 0, 0, True
        skill_steps = 0
        skill_now = 0
        start_time = time.time()
        self.all_skill_rewards = []

        # OmegaConf can be slow, so assigning to local variables.
        num_train_steps = self.cfg.num_train_steps
        num_steps_per_skill = self.cfg.num_steps_per_skill
        blocks_to_remove_at_once = self.cfg.blocks_to_remove_at_once
        num_seed_steps = self.cfg.num_seed_steps
        update_per_step = self.cfg.updates_per_step

        # Initialize the reward computation stacks
        while self.step < num_train_steps:
            if done:
                self._do_logging(episode, episode_reward, start_time)
                start_time = time.time()

                # We have to collect trajectories before we do any reset.
                if skill_steps >= num_steps_per_skill[skill_now]:
                    print("Collecting trajectories")
                    # Here, we reset the collected trajectories buffer since we want
                    # to collect fresh behavior in a potentially new environment.
                    self.reward_module.reset_collected_trajectories()
                    for i in range(skill_now):
                        self.collect_trajectories(skill_idx=i)
                        self.reward_module.save_collected_trajectories()
                    print("Done collecting trajectories")

                    self.evaluate(record=True)

                    skill_now += 1
                    skill_steps = 0
                    # purge half examples from the replay buffer
                    self.agent.add_new_skill(num_steps_per_skill[skill_now])
                    self.reward_module.add_new_skill(num_steps_per_skill[skill_now])
                    self.replay_buffer.purge_frac(0.5)
                    if hasattr(self.env, "add_new_skill"):
                        self.env.add_new_skill()

                obs = self.env.reset()
                transformed_obs = self.transform_obs(obs)
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                self.episode = episode
                self.logger.log("train/episode", episode, self.step)

            if (self._next_block_to_remove < 0) and (
                (self.step + 1) % self._block_removal_steps == 0
            ):
                # Remove a block from env by burying it.
                print("---REMOVING BLOCKS---")
                for _ in range(int(blocks_to_remove_at_once)):
                    self.env.model.body_pos[self._next_block_to_remove, -1] = -4.0
                    self._next_block_to_remove -= 1

            # sample action for data collection
            if skill_steps < num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True,)
                # run training update
                for _ in range(update_per_step):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)
            transformed_next_obs = self.transform_obs(next_obs)

            episode_step += 1
            self.step += 1
            skill_steps += 1

            if episode_step > self.max_episode_steps:
                done = True
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            latent_reward = 0
            self.reward_module.add_current(
                transformed_obs, obs, transformed_next_obs, next_obs, episode_step, done
            )
            self.replay_buffer.add(
                obs,
                transformed_obs,
                action,
                latent_reward,
                next_obs,
                transformed_next_obs,
                done,
                done_no_max,
            )
            obs = next_obs.copy()
            transformed_obs = transformed_next_obs.copy()

        print("Done training")
        self.evaluate(True)

    def transform_obs(self, obs):
        if self.obs_transform:
            return self.obs_transform(self.env, obs)
        return obs

    def _setup_transformed_obses(self):
        self.env_prefix = re.split("-|_", self.env_name)[0]  # match either - or _
        self.obs_transform = OBS_TRANSFORMS.get(self.env_prefix, obs_transform_default)
        sample_transformed_obs = self.obs_transform(self.env, self.env.reset())
        self.cfg.transformed_obs_shape = sample_transformed_obs.shape

        self.env._max_episode_steps = self.cfg.max_episode_steps
        self.max_episode_steps = self.cfg.max_episode_steps
        self.collected_trajectories = int(self.cfg.collected_trajectories)

        self.cfg.agent.obs_dim = self.env.observation_space.shape[0]
        self.cfg.agent.t_obs_dim = sample_transformed_obs.shape[0]
        self.cfg.agent.action_dim = self.env.action_space.shape[0]
        self.cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]

    def _do_logging(self, episode, episode_reward, start_time):
        if self.step > 0:
            self.logger.log("train/duration", time.time() - start_time, self.step)
            self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

        # evaluate agent periodically
        if self.step > 0 and episode % self.cfg.eval_frequency == 0:
            self.logger.log("eval/episode", episode, self.step)

        self.logger.log("train/episode_reward", episode_reward, self.step)

    def _setup_number_of_skills(self):
        if isinstance(self.cfg.num_steps_per_skill, (int, float)):
            # Set the number of skills
            steps_per_skill = int(self.cfg.num_steps_per_skill)
            self.cfg.total_skills = int(self.cfg.num_train_steps // steps_per_skill)
            self.cfg.num_steps_per_skill = [
                steps_per_skill for _ in range(self.cfg.total_skills)
            ]
        else:
            self.cfg.num_steps_per_skill = list(self.cfg.num_steps_per_skill)
            self.cfg.total_skills = len(self.cfg.num_steps_per_skill)

    def _setup_buffers(self):
        self.replay_buffer = buffers.ReplayBuffer(
            self.env.observation_space.shape,
            self.cfg.transformed_obs_shape,
            self.env.action_space.shape,
            int(self.cfg.replay_buffer_capacity),
            self.device,
        )

        self.reward_module = buffers.RewardBuffer(
            self.cfg.transformed_obs_shape,
            self.max_episode_steps,
            int(self.cfg.saved_latent_per_skill),
            int(self.cfg.total_skills),
            max_running_obses=int(self.cfg.max_running_obses),
            slow_update_coeff=int(self.cfg.slow_update_coeff),
            device=self.device,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
        )
        self.agent.register_reward_module(self.reward_module)
        self.reward_module.register_logger(self.logger)

    def _setup_video_recording(self):
        self.rec_transform = RECORD_TRANSFORMS.get(
            self.env_prefix, record_transform_default
        )
        sample_record_obs = self.rec_transform(self.env, self.env.reset())
        self.video_recorder = utils.VideoRecorderWithStates(
            self.work_dir if self.cfg.save_video else None,
            fps=30,
            obs_transforms=self.rec_transform,
            num_transforms=sample_record_obs.shape[0],
        )

    def _setup_blocks_if_needed(self):
        if self.env_name == "Ant-block":
            self._next_block_to_remove = -1
        else:
            self._next_block_to_remove = 0
        self._block_removal_steps = int(self.cfg.num_train_steps) // (
            int(self.cfg.num_blocks) // int(self.cfg.blocks_to_remove_at_once)
        )


@hydra.main(config_path="config", config_name="train")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
