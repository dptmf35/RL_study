#!/usr/bin/env python3
"""
Evaluation Script for Stretch Robot Grasping

Loads trained SAC model and evaluates performance with visualization.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.stretch_grasp_env import StretchGraspEnv


def evaluate(args):
    """Run evaluation."""
    print(f"Loading model from: {args.model_path}")

    # Create environment
    env = StretchGraspEnv(
        render_mode="human" if args.render else None,
        max_episode_steps=args.max_steps,
        reward_type="dense",
        randomize_cube=True,
    )

    # Wrap for VecNormalize
    env = DummyVecEnv([lambda: env])

    # Load normalization stats
    vec_normalize_path = Path(args.model_path).parent / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        print(f"Loading normalization stats from: {vec_normalize_path}")
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: No VecNormalize stats found")

    # Load model
    model = SAC.load(args.model_path, env=env)
    print("Model loaded successfully")

    # Run episodes
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
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1

            if args.render:
                env.envs[0].render()
                if args.slow_motion:
                    time.sleep(0.02)

            if done[0]:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if info[0].get("task_success", False):
            success_count += 1
            print(f"Episode {episode + 1}: SUCCESS - Reward: {episode_reward:.2f}, Length: {episode_length}")
        else:
            print(f"Episode {episode + 1}: FAILED  - Reward: {episode_reward:.2f}, Length: {episode_length}")

    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes: {args.n_episodes}")
    print(f"Success Rate: {success_count}/{args.n_episodes} ({100*success_count/args.n_episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print("=" * 50)

    env.close()


def demo_random(args):
    """Demo with random actions."""
    print("Running random action demo...")

    env = StretchGraspEnv(
        render_mode="human",
        max_episode_steps=args.max_steps,
        randomize_cube=True,
    )

    for episode in range(args.n_episodes):
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1}")
        print(f"Initial distance to cube: {info['distance_to_cube']:.3f}m")

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
    parser = argparse.ArgumentParser(description="Evaluate Stretch Grasping Agent")

    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to trained model")
    parser.add_argument("--n-episodes", type=int, default=10,
                       help="Number of episodes (default: 10)")
    parser.add_argument("--max-steps", type=int, default=300,
                       help="Max steps per episode (default: 300)")
    parser.add_argument("--render", action="store_true", default=True,
                       help="Render visualization")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic actions")
    parser.add_argument("--slow-motion", action="store_true",
                       help="Slow motion visualization")
    parser.add_argument("--random", action="store_true",
                       help="Use random actions (no model)")

    args = parser.parse_args()

    if args.no_render:
        args.render = False

    if not args.random and args.model_path is None:
        parser.error("--model-path required unless --random specified")

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.random:
        demo_random(args)
    else:
        evaluate(args)
