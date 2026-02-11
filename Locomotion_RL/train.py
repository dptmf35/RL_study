"""
PPO Training Script for Unitree Go2 Locomotion.

Usage:
    python train.py                              # Train on flat ground (5M steps)
    python train.py --terrain                    # Train on terrain
    python train.py --terrain --difficulty 0.7   # Harder terrain
    python train.py --timesteps 1000000          # Custom timestep count
    python train.py --resume checkpoints/best_model.zip  # Resume training
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

sys.path.insert(0, os.path.dirname(__file__))
from envs.go2_env import Go2Env
from envs.go2_terrain_env import Go2TerrainEnv
from configs.go2_config import TRAIN_CONFIG


def make_env(env_config, seed=0, use_terrain=False, terrain_config=None):
    """Create a Go2 environment factory."""
    def _init():
        if use_terrain:
            t_cfg = terrain_config or {}
            env = Go2TerrainEnv(
                difficulty=t_cfg.get("difficulty", 0.5),
                terrain_seed=seed,  # each env gets unique terrain
                **env_config,
            )
        else:
            env = Go2Env(**env_config)
        env.reset(seed=seed)
        return env
    return _init


def train(args):
    config = TRAIN_CONFIG.copy()
    total_timesteps = args.timesteps or config["total_timesteps"]
    n_envs = args.n_envs or config["n_envs"]
    use_terrain = args.terrain
    terrain_config = config.get("terrain_env", {})

    if args.difficulty is not None:
        terrain_config["difficulty"] = args.difficulty

    # Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "terrain" if use_terrain else "flat"
    run_name = f"go2_{mode}_{timestamp}"
    log_dir = os.path.join(config["log_dir"], run_name)
    model_dir = os.path.join(config["model_dir"], run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"=== Go2 Locomotion Training ===")
    print(f"Mode            : {mode}")
    if use_terrain:
        print(f"Terrain difficulty: {terrain_config.get('difficulty', 0.5)}")
    print(f"Total timesteps : {total_timesteps:,}")
    print(f"Parallel envs   : {n_envs}")
    print(f"Log dir         : {log_dir}")
    print(f"Model dir       : {model_dir}")
    print()

    # Create vectorized training environments
    env_config = config["env"]
    train_envs = SubprocVecEnv(
        [make_env(env_config, seed=i, use_terrain=use_terrain,
                  terrain_config=terrain_config) for i in range(n_envs)]
    )
    train_envs = VecNormalize(
        train_envs,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Create evaluation environment
    eval_env = SubprocVecEnv(
        [make_env(env_config, seed=100, use_terrain=use_terrain,
                  terrain_config=terrain_config)]
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(config["eval_freq"] // n_envs, 1),
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(config["save_freq"] // n_envs, 1),
        save_path=model_dir,
        name_prefix="go2_ppo",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # PPO model
    ppo_config = config["ppo"]
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_envs)
        vecnorm_path = args.resume.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            train_envs = VecNormalize.load(vecnorm_path, train_envs)
            print(f"Loaded VecNormalize from: {vecnorm_path}")
    else:
        model = PPO(
            "MlpPolicy",
            train_envs,
            learning_rate=ppo_config["learning_rate"],
            n_steps=ppo_config["n_steps"],
            batch_size=ppo_config["batch_size"],
            n_epochs=ppo_config["n_epochs"],
            gamma=ppo_config["gamma"],
            gae_lambda=ppo_config["gae_lambda"],
            clip_range=ppo_config["clip_range"],
            ent_coef=ppo_config["ent_coef"],
            vf_coef=ppo_config["vf_coef"],
            max_grad_norm=ppo_config["max_grad_norm"],
            policy_kwargs=ppo_config["policy_kwargs"],
            verbose=1,
            tensorboard_log=log_dir,
            device="auto",
        )

    # Configure logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    print(f"\nPolicy architecture: {model.policy}")
    print(f"Device: {model.device}")
    print(f"\nStarting training...\n")

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    train_envs.save(f"{final_path}_vecnormalize.pkl")
    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_path}.zip")
    print(f"VecNormalize saved to: {final_path}_vecnormalize.pkl")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Go2 Locomotion Policy")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume training from")
    parser.add_argument("--terrain", action="store_true", help="Train on terrain (heightfield)")
    parser.add_argument("--difficulty", type=float, default=None, help="Terrain difficulty 0.0-1.0")
    args = parser.parse_args()
    train(args)
