import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import VecVideoRecorder


def post_evaluate(
    vec_env,
    best_model_save_path,
    num_envs,
    n_episodes,
    log_id,
    log_path,
    render,
    enable_recording,
    recording_interval,
    recording_length,
    recording_dir,
    z_log_freq=0,
):
    log_path += f"/{log_id}"
    if enable_recording:
        vec_env = VecVideoRecorder(
            vec_env,
            log_path + recording_dir,
            record_video_trigger=lambda x: x % recording_interval == 0,
            video_length=recording_length,
        )

    # observation reset
    obs = vec_env.reset()

    # best model loading
    model = ...

    # Cell and hidden state of the RNN
    rnn_states = None
    num_envs = vec_env.num_envs

    # Episode start signals are used to reset the rnn states
    episode_starts = np.ones((num_envs,), dtype=bool)

    episode_cnt = 0
    step = 0
    rewards_ = np.array([])
    z_ = np.array([])
    alpha_ = np.array([])

    while episode_cnt < n_episodes:

        action, rnn_states = model.predict(
            obs, state=rnn_states, episode_start=episode_starts, deterministic=True
        )
        clipped_action = np.clip(
            action, vec_env.action_space.low, vec_env.action_space.high
        )

        # Note: vectorized environment resets automatically
        try:
            z_log_trigger = step % z_log_freq == 0
        except ZeroDivisionError:
            z_log_trigger = False

        if z_log_trigger:
            z = model.policy.mlp_extractor.policy_net["nm_net_actor"].nmdict[
                "nm_z_actor"
            ]
            z_ = np.append(z_, z.cpu().numpy())
            alpha_ = np.append(alpha_, vec_env.alpha)
            z_log_trigger = False

        obs, _, done, info = vec_env.step(clipped_action)

        if render:
            vec_env.render(render_mode="human")

        episode_starts = done
        episode_cnt += done.sum()

        np.append(rewards_, info[0]["reward"])

    np.save(log_path + "/rewards.npy", rewards_)
    np.save(log_path + "/z.npy", z_)
    np.save(log_path + "/alpha.npy", alpha_)
