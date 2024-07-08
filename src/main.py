from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize

from omegaconf import DictConfig, OmegaConf
from sb3_contrib import RecurrentPPO
import hydra
import time
import wandb

from utils import create_env, get_class_from_path
from post_evaluation import post_evaluate
import environments

# import pprint

# import torch
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

import warnings
warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path="./cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    agent_cfg = cfg.agent
    train_cfg = cfg.train
    task_cfg = cfg.task

    # Create environment
    vec_env = create_env(
        env_id=cfg.task_name,
        env_wrappers=task_cfg.wrappers.wrapper_list,
        env_kwargs=task_cfg.env.params,
        num_envs=task_cfg.env.num_envs,
        seed=cfg.seed,
        norm_obs=train_cfg.norm_obs,
        norm_rews=train_cfg.norm_rews,
        gamma=train_cfg.gamma,
    )
    if cfg.eval:
        vec_eval_env = create_env(
            env_id=cfg.task_name,
            env_wrappers=task_cfg.wrappers.wrapper_list,
            env_kwargs=task_cfg.env.params,
            num_envs=task_cfg.env.num_envs,
            seed=cfg.seed,
            norm_obs=train_cfg.norm_obs,
            norm_rews=train_cfg.norm_rews,
            gamma=train_cfg.gamma,
        )

    # Wandb
    run_name = cfg.wandb_name + "_" + time.strftime("%d-%m-%y-%H-%M-%S")
    train_cfg_dict = OmegaConf.to_container(train_cfg, resolve=True)
    run = wandb.init(
        mode="online" if cfg.wandb_activate else "disabled",
        project=cfg.wandb_project,
        name=run_name,
        config=train_cfg_dict,
        sync_tensorboard=True,
    )

    callback = []
    if cfg.wandb_activate:
        callback.append(
            WandbCallback(
                gradient_save_freq=cfg.save_grad_freq,
                model_save_freq=cfg.save_freq,
                model_save_path=cfg.save_dir,
                verbose=2,
            ),
        )

    # Video recording
    if cfg.enable_recording:
        vec_env = VecVideoRecorder(
            vec_env,
            cfg.recording_dir + f"/{run.id}",
            record_video_trigger=lambda x: x % cfg.recording_interval == 0,
            video_length=cfg.recording_length,
        )
        if cfg.eval:
            vec_eval_env = VecVideoRecorder(
                vec_eval_env,
                cfg.recording_dir + f"/{run.id}",
                record_video_trigger=lambda x: x % cfg.recording_interval == 0,
                video_length=cfg.recording_length,
            )
            callback.append(
                EvalCallback(
                    eval_env=vec_eval_env,
                    n_eval_episodes=cfg.eval_n_episodes,
                    eval_freq=max(cfg.eval_freq // task_cfg.env.num_envs, 1),
                    log_path=cfg.eval_log_path,
                    best_model_save_path=cfg.eval_best_model_save_path,
                    deterministic=cfg.eval_deterministic,
                    render=cfg.eval_render,
                    verbose=cfg.eval_verbose,
                )
            )

    # create policy
    if agent_cfg.name == "nmn":
        agent_cfg.policy.kwargs.rnn_input_dim = (
            vec_env.observation_space.shape[0] - vec_env.observation_space.init_obs_dim
        )
        agent_cfg.mlp.features_dim = vec_env.observation_space.init_obs_dim

    # convert path to class for activations
    policy_kwargs = OmegaConf.to_container(
        agent_cfg.policy.kwargs,
        resolve=True,
    )

    policy_kwargs["rnn_type"] = get_class_from_path(policy_kwargs["rnn_type"])
    policy_kwargs["mlp_kwargs"]["activation_cls"] = get_class_from_path(
        policy_kwargs["mlp_kwargs"]["activation_cls"]
    )
    if agent_cfg.name == "nmn":
        policy_kwargs["nm_activation_cls"] = get_class_from_path(
            policy_kwargs["nm_activation_cls"]
        )
        policy_kwargs["mlp_kwargs"]["nm_activation_cls"] = get_class_from_path(
            policy_kwargs["mlp_kwargs"]["nm_activation_cls"]
        )

    # train
    with run:
        model = RecurrentPPO(
            policy=get_class_from_path(agent_cfg.policy.cls),
            policy_kwargs=policy_kwargs,
            env=vec_env,
            learning_rate=train_cfg.learning_rate,  # can be float or Schedule
            n_steps=train_cfg.n_steps,
            batch_size=train_cfg.batch_size,
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            clip_range=train_cfg.clip_range,
            clip_range_vf=None,
            normalize_advantage=train_cfg.norm_adv,
            ent_coef=train_cfg.ent_coef,
            vf_coef=train_cfg.vf_coef,
            max_grad_norm=1.,
            use_sde=train_cfg.use_sde,
            use_beta=train_cfg.use_beta,
            sde_sample_freq=train_cfg.sde_sample_freq,
            target_kl=None, 
            stats_window_size=100,
            seed=None,
            verbose=0,
            tensorboard_log=cfg.tensorboard_log_dir + f"/{run.id}",
            device=cfg.rl_device,
        )

        # print(model.policy)
        model.learn(
            total_timesteps=train_cfg.total_n_steps,
            callback=callback,
            progress_bar=True,
        )

        post_eval_cfg = cfg.post_eval
        if post_eval_cfg.enable:
            post_evaluate(
                create_env(
                    env_id=cfg.task_name,
                    env_wrappers=task_cfg.wrappers.wrapper_list,
                    env_kwargs=task_cfg.env.post_eval.params,
                    num_envs=post_eval_cfg.num_envs,
                    seed=post_eval_cfg.seed,
                    normalize=train_cfg.normalize_input,
                    gamma=train_cfg.gamma,
                ),
                cfg.eval_best_model_save_path,
                run.id,
                post_eval_cfg.n_episodes,
                post_eval_cfg.log_path,
                post_eval_cfg.render,
                post_eval_cfg.enable_recording,
                post_eval_cfg.recording_interval,
                post_eval_cfg.recording_length,
                post_eval_cfg.recording_dir,
                post_eval_cfg.z_log_freq,
            )


if __name__ == "__main__":
    main()
