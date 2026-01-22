#!/usr/bin/env python3
"""
SAC + HER Training Script for UR5e + Robotiq 2F85 Pick and Place Environment

SAC with Hindsight Experience Replay is the recommended algorithm for
sparse-reward manipulation tasks (as used in Gymnasium-Robotics Fetch environments).

Usage:
    python scripts/train_sac_her_robotiq.py --task-mode reach --easy-mode
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

# HER imports - handle different SB3 versions
try:
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
except ImportError:
    from stable_baselines3.her import HerReplayBuffer, GoalSelectionStrategy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.ur5e_robotiq_goal_env import UR5eRobotiqGoalEnv


def make_env(seed: int = 0, task_mode: str = "reach", easy_mode: bool = False):
    """Create a goal-conditioned environment."""
    def _init():
        env = UR5eRobotiqGoalEnv(
            render_mode=None,
            max_episode_steps=100,
            task_mode=task_mode,
            easy_mode=easy_mode,
        )
        env.reset(seed=seed)
        return env
    return _init


def train(args):
    """Main training function."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    easy_str = "_easy" if args.easy_mode else ""
    run_name = f"sac_her_ur5e_robotiq_{args.task_mode}{easy_str}_{timestamp}"

    log_dir = Path(args.log_dir) / run_name
    model_dir = Path(args.model_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training run: {run_name}")
    print(f"Task mode: {args.task_mode}")
    print(f"Easy mode: {args.easy_mode}")
    print(f"Using: SAC + HER (Hindsight Experience Replay)")

    # Create environment (HER requires non-vectorized env for training)
    env = make_env(args.seed, args.task_mode, args.easy_mode)()

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(args.seed + 1000, args.task_mode, args.easy_mode)])

    # SAC + HER
    model = SAC(
        policy="MultiInputPolicy",  # Required for Dict observation space
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,  # Number of virtual goals per transition
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,  # Sample future goals
        ),
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.05,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        device=args.device,
    )

    print(f"\nModel: SAC with HER")
    print(f"Goal selection strategy: FUTURE")

    # Callbacks
    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(model_dir),
        name_prefix="sac_her_ur5e_robotiq",
    )
    callbacks.append(checkpoint_callback)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
    )
    callbacks.append(eval_callback)

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
    model.save(str(model_dir / "final_model"))
    print(f"\nFinal model saved to: {model_dir / 'final_model'}")

    env.close()
    eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC+HER for UR5e Robotiq")

    parser.add_argument("--task-mode", type=str, default="reach",
                        choices=["reach", "pick", "pick_place"])
    parser.add_argument("--easy-mode", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
