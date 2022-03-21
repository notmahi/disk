"""
Different encoding priors that are supplied to the incremental agents. The
prior used here are simple, in two ways. First, the observation that the agent
uses to learn is kept the same as before. But, the transformed observation
that is used to form the entropy-based reward is modified in each case. One
rule of thumb that we use is this: any extra dimension that is added as the
prior for the agent reward must be added to the front of standard observation.
"""

import numpy as np


def obs_transform_quad(env, obs=None):
    t_obs = env.env._env._physics._named.data.xpos["torso"][:2]
    return np.copy(t_obs)


def obs_transform_ant(env, obs=None):
    t_obs = env.get_body_com("torso")[:2].copy()
    return t_obs


def obs_transform_humanoid(env, obs=None):
    t_obs = env.get_body_com("torso")[:2].copy()
    return t_obs


def obs_transform_hopper(env, obs=None):
    data = env.get_body_com("torso")[0]
    return np.array([data, 0])


def record_transform_hopper(env, obs=None):
    xpos = env.get_body_com("torso")[0]
    zpos = env.get_body_com("torso")[2]
    return np.array([xpos, zpos])


def obs_transform_cheetah(env, obs=None):
    data = env.env._env._physics._named.data

    return np.array(
        [
            data.xpos["torso"][0],
            data.xpos["torso"][2],
        ]  # X position  # Z position
    )


def obs_transform_swimmer(env, obs=None):
    physics = env.env._env._physics
    data = physics._named.data
    center_of_mass = data.subtree_com["head"]
    return np.array(
        [
            center_of_mass[0],
            center_of_mass[1],
        ]
    )  # X position  # Y position


def obs_transform_gym_swimmer(env, obs=None):
    position = env.sim.data.qpos.flat.copy()
    return position[0:2]


def obs_transform_gym_swimmer_com(env, obs=None):
    position = env.sim.data.subtree_com.flat[0:2]
    return position.copy()


def obs_transform_default(env, obs):
    return obs


def record_transform_default(env, obs):
    return np.array([])


OBS_TRANSFORMS = {
    "quadruped": obs_transform_quad,
    "Ant": obs_transform_ant,
    "cheetah": obs_transform_cheetah,
    "HalfCheetah": record_transform_hopper,
    "swimmer": obs_transform_swimmer,
    "Swimmer": obs_transform_gym_swimmer_com,
    "Hopper": record_transform_hopper,
    "Humanoid": obs_transform_humanoid,
}

RECORD_TRANSFORMS = {
    "quadruped": obs_transform_quad,
    "Ant": obs_transform_ant,
    "cheetah": obs_transform_cheetah,
    "HalfCheetah": record_transform_hopper,
    "swimmer": obs_transform_swimmer,
    "Swimmer": obs_transform_gym_swimmer_com,
    "Hopper": record_transform_hopper,
    "Humanoid": obs_transform_humanoid,
}
