#!/usr/bin/env python3
"""
Evaluation and Visualization Script for UR5e Pick and Place

This script loads a trained PPO model and runs evaluation episodes
with visualization.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.ur5e_pick_place_env import UR5ePickPlaceEnv


def evaluate(args):
    """Run evaluation with visualization."""
    print(f"Loading model from: {args.model_path}")

    # Create environment
    env = UR5ePickPlaceEnv(
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps,
        reward_type="dense",
        randomize_cube=True,
        randomize_target=False,
    )

    # Wrap in DummyVecEnv for compatibility with VecNormalize
    env = DummyVecEnv([lambda: env])

    # Load VecNormalize stats if available
    vec_normalize_path = Path(args.model_path).parent / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        print(f"Loading normalization stats from: {vec_normalize_path}")
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: No VecNormalize stats found, using raw observations")

    # Load model
    model = PPO.load(args.model_path, env=env)
    print("Model loaded successfully")

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print(f"\nRunning {args.n_episodes} evaluation episodes...")

    for episode in range(args.n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get action from policy
            if args.deterministic:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=False)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1

            # Render
            if args.render:
                env.envs[0].render()
                if args.slow_motion:
                    time.sleep(0.02)

            # Check for done
            if done[0]:
                break

        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check success
        final_info = info[0]
        if final_info.get("task_success", False):
            success_count += 1
            print(f"Episode {episode + 1}: SUCCESS - Reward: {episode_reward:.2f}, Length: {episode_length}")
        else:
            print(f"Episode {episode + 1}: FAILED  - Reward: {episode_reward:.2f}, Length: {episode_length}")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes: {args.n_episodes}")
    print(f"Success Rate: {success_count}/{args.n_episodes} ({100*success_count/args.n_episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print("=" * 50)

    env.close()

    return {
        "success_rate": success_count / args.n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }


def demo_random(args):
    """Run demo with random actions."""
    print("Running random action demo...")

    env = UR5ePickPlaceEnv(
        render_mode="human",
        max_episode_steps=args.max_steps,
        reward_type="dense",
        randomize_cube=True,
    )

    for episode in range(args.n_episodes):
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1}")
        print(f"Initial cube position: {info['distance_to_cube']:.3f}m from gripper")

        episode_reward = 0
        for step in range(args.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            env.render()

            if args.slow_motion:
                time.sleep(0.02)

            if terminated or truncated:
                print(f"Episode ended at step {step + 1}")
                print(f"Total reward: {episode_reward:.2f}")
                print(f"Success: {info.get('task_success', False)}")
                break

    env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize trained UR5e Pick and Place agent"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (required for evaluation)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Render visualization (default: True)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)",
    )
    parser.add_argument(
        "--slow-motion",
        action="store_true",
        help="Run in slow motion for better visualization",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Run with random actions (no model needed)",
    )

    args = parser.parse_args()

    if args.no_render:
        args.render = False

    if not args.random and args.model_path is None:
        parser.error("--model-path is required unless --random is specified")

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.random:
        demo_random(args)
    else:
        evaluate(args)
