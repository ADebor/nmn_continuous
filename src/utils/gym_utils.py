import gymnasium as gym

from hydra.utils import instantiate
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecNormalize,
    VecMonitor,
)


def make_env(
    env_id: str,
    wrapper_classes: list,
    env_kwargs: dict = None,
):
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)

    if wrapper_classes is not None:
        for wrapper_class in wrapper_classes:
            env = instantiate(wrapper_class, env=env)
        return env
    else:
        return env


def create_env(env_id, env_wrappers, env_kwargs, num_envs, seed, norm_obs, norm_rews, gamma):
   
    vec_env = make_vec_env(
        make_env,
        n_envs=num_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "env_id": env_id,
            "env_kwargs": env_kwargs,
            "wrapper_classes": env_wrappers,
        },
    )

    if norm_obs or norm_rews:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=norm_obs,
            norm_reward=norm_rews,
            gamma=gamma,
        )
    
    return VecMonitor(vec_env)
