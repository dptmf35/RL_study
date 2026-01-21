#!/usr/bin/env python3
"""
SAC Training Script for Stretch Robot Grasping

Uses Soft Actor-Critic (SAC) algorithm from Stable-Baselines3.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.stretch_grasp_env import StretchGraspEnv


class SuccessRateCallback(BaseCallback):
    """Callback to track success rate."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_count = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][idx]
                self.episode_count += 1

                if info.get("task_success", False):
                    self.success_count += 1

                if self.episode_count % 100 == 0:
                    success_rate = self.success_count / self.episode_count
                    self.logger.record("custom/success_rate", success_rate)
                    self.logger.record("custom/total_episodes", self.episode_count)

        return True


def make_env(rank: int, seed: int = 0, randomize_cube: bool = True):
    """Create environment."""
    def _init():
        env = StretchGraspEnv(
            render_mode=None,
            max_episode_steps=300,
            reward_type="dense",
            randomize_cube=randomize_cube,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args):
    """Main training function."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_stretch_grasp_{timestamp}"

    log_dir = Path(args.log_dir) / run_name
    model_dir = Path(args.model_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training run: {run_name}")
    print(f"Algorithm: SAC")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")

    # Create environment
    if args.n_envs > 1:
        env = SubprocVecEnv(
            [make_env(i, args.seed, not args.no_randomize) for i in range(args.n_envs)]
        )
    else:
        env = DummyVecEnv([make_env(0, args.seed, not args.no_randomize)])

    # Normalize observations
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Evaluation environment
    eval_env = DummyVecEnv([make_env(0, args.seed + 1000, not args.no_randomize)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    # SAC hyperparameters
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            qf=[256, 256],
        ),
        activation_fn=torch.nn.ReLU,
    )

    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    print(f"\nModel architecture:")
    print(model.policy)

    # Callbacks
    callbacks = []

    # Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path=str(model_dir),
        name_prefix="sac_stretch",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation
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

    # Success rate tracking
    success_callback = SuccessRateCallback()
    callbacks.append(success_callback)

    # Train
    print(f"\nStarting training for {args.total_timesteps} timesteps...")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted")

    # Save final model
    final_path = model_dir / "final_model"
    model.save(str(final_path))
    env.save(str(model_dir / "vec_normalize.pkl"))

    print(f"\nTraining complete!")
    print(f"Final model: {final_path}")

    env.close()
    eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC for Stretch Grasping")

    # Environment
    parser.add_argument("--no-randomize", action="store_true",
                       help="Disable cube position randomization")

    # Training
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                       help="Total training timesteps (default: 500K)")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    # SAC hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
                       help="Replay buffer size (default: 1M)")
    parser.add_argument("--learning-starts", type=int, default=10_000,
                       help="Steps before learning starts (default: 10K)")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size (default: 256)")
    parser.add_argument("--tau", type=float, default=0.005,
                       help="Soft update coefficient (default: 0.005)")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor (default: 0.99)")
    parser.add_argument("--train-freq", type=int, default=1,
                       help="Training frequency (default: 1)")
    parser.add_argument("--gradient-steps", type=int, default=1,
                       help="Gradient steps per update (default: 1)")
    parser.add_argument("--ent-coef", type=str, default="auto",
                       help="Entropy coefficient (default: auto)")

    # Logging
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Log directory (default: logs)")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Model directory (default: models)")
    parser.add_argument("--save-freq", type=int, default=50_000,
                       help="Checkpoint frequency (default: 50K)")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                       help="Evaluation frequency (default: 10K)")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                       help="Evaluation episodes (default: 10)")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device (default: auto)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
