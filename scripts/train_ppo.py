#!/usr/bin/env python3
"""
PPO Training Script for UR5e Pick and Place Environment

This script trains a PPO agent using Stable-Baselines3 to solve
the UR5e pick and place task.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.ur5e_pick_place_env import UR5ePickPlaceEnv


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Log episode statistics when episode ends
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][idx]
                self.episode_count += 1

                if info.get("task_success", False):
                    self.success_count += 1

                # Log success rate every 100 episodes
                if self.episode_count % 100 == 0:
                    success_rate = self.success_count / self.episode_count
                    self.logger.record("custom/success_rate", success_rate)
                    self.logger.record("custom/total_episodes", self.episode_count)

        return True


def make_env(rank: int, seed: int = 0, reward_type: str = "dense"):
    """Create a wrapped, monitored environment."""

    def _init():
        env = UR5ePickPlaceEnv(
            render_mode=None,
            max_episode_steps=500,
            reward_type=reward_type,
            randomize_cube=True,
            randomize_target=False,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def train(args):
    """Main training function."""
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_ur5e_{timestamp}"

    log_dir = Path(args.log_dir) / run_name
    model_dir = Path(args.model_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training run: {run_name}")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")

    # Create vectorized environment
    if args.n_envs > 1:
        env = SubprocVecEnv(
            [make_env(i, args.seed, args.reward_type) for i in range(args.n_envs)]
        )
    else:
        env = DummyVecEnv([make_env(0, args.seed, args.reward_type)])

    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(0, args.seed + 1000, args.reward_type)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward for evaluation
        clip_obs=10.0,
        training=False,
    )

    # Define network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Policy network
            vf=[256, 256, 128],  # Value network
        ),
        activation_fn=torch.nn.ReLU,
    )

    # Create PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    print(f"\nModel architecture:")
    print(model.policy)

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=str(model_dir),
        name_prefix="ppo_ur5e",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Custom tensorboard callback
    tensorboard_callback = TensorboardCallback()
    callbacks.append(tensorboard_callback)

    # Train the agent
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print(f"Using {args.n_envs} parallel environments")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))
    env.save(str(model_dir / "vec_normalize.pkl"))

    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"VecNormalize stats saved to: {model_dir / 'vec_normalize.pkl'}")

    # Cleanup
    env.close()
    eval_env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for UR5e Pick and Place"
    )

    # Environment
    parser.add_argument(
        "--reward-type",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="Reward type (default: dense)",
    )

    # Training
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Total training timesteps (default: 2M)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per environment per update (default: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size (default: 256)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of epochs per update (default: 10)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda (default: 0.95)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range (default: 0.2)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Max gradient norm (default: 0.5)",
    )

    # Logging and saving
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for tensorboard logs (default: logs)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for saved models (default: models)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Model checkpoint frequency (default: 50000)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=20_000,
        help="Evaluation frequency (default: 20000)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
