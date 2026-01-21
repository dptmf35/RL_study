#!/usr/bin/env python3
"""
Evaluation Script for UR5e + Robotiq 2F85 Pick and Place Environment

This script evaluates trained models or runs demonstrations.
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

from envs.ur5e_robotiq_pick_place_env import UR5eRobotiqPickPlaceEnv


def evaluate_model(args):
    """Evaluate a trained model."""
    print(f"Loading model from: {args.model_path}")
    
    # Load model
    model = PPO.load(args.model_path)
    
    # Try to load VecNormalize stats
    vec_normalize_path = Path(args.model_path).parent.parent / "vec_normalize.pkl"
    if not vec_normalize_path.exists():
        vec_normalize_path = Path(args.model_path).parent / "vec_normalize.pkl"
    
    # Create environment
    env = UR5eRobotiqPickPlaceEnv(
        render_mode="human" if not args.no_render else None,
        max_episode_steps=500,
        reward_type="dense",
        randomize_cube=not args.static_cube,
        randomize_target=False,
        task_mode=args.task_mode,
        easy_mode=args.easy_mode,
    )
    
    # Wrap with VecNormalize if available
    if vec_normalize_path.exists():
        print(f"Loading VecNormalize stats from: {vec_normalize_path}")
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: VecNormalize stats not found, using unnormalized observations")
        env = DummyVecEnv([lambda: env])
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\nRunning {args.n_episodes} episodes...")
    
    for episode in range(args.n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=args.deterministic)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            # Slow motion if requested
            if args.slow_motion:
                time.sleep(0.05)
            
            # Check if done
            if done[0]:
                break
        
        # Log episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info[0].get("task_success", False):
            success_count += 1
            status = "SUCCESS"
        else:
            status = "FAILED"
        
        print(f"Episode {episode+1}/{args.n_episodes}: "
              f"Reward={episode_reward:.2f}, Length={episode_length}, {status}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({args.n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
    print(f"Success rate: {success_count}/{args.n_episodes} ({100*success_count/args.n_episodes:.1f}%)")
    print(f"{'='*60}\n")
    
    env.close()


def demo_random(args):
    """Run random action demonstration."""
    print("Running random action demo...")
    
    env = UR5eRobotiqPickPlaceEnv(
        render_mode="human" if not args.no_render else None,
        max_episode_steps=500,
        randomize_cube=not args.static_cube,
        task_mode=args.task_mode,
        easy_mode=args.easy_mode,
    )
    
    for episode in range(args.n_episodes):
        obs, info = env.reset()
        print(f"\nEpisode {episode+1}/{args.n_episodes}")
        episode_reward = 0
        
        for step in range(500):
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            env.render()
            
            if args.slow_motion:
                time.sleep(0.05)
            
            if terminated or truncated:
                status = "SUCCESS" if info.get("task_success", False) else "FAILED"
                print(f"  Episode ended at step {step+1}, Reward={episode_reward:.2f}, {status}")
                break
    
    env.close()


def demo_static(args):
    """Run static robot demonstration (no movement)."""
    print("Running static demo (robot stays still)...")
    
    env = UR5eRobotiqPickPlaceEnv(
        render_mode="human" if not args.no_render else None,
        max_episode_steps=500,
        randomize_cube=not args.static_cube,
    )
    
    for episode in range(args.n_episodes):
        obs, info = env.reset()
        print(f"\nEpisode {episode+1}/{args.n_episodes} - Robot staying still")
        
        for step in range(200):
            # No action (zeros)
            action = np.zeros(env.action_space.shape)
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if args.slow_motion:
                time.sleep(0.05)
            
            if terminated or truncated:
                break
    
    env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate UR5e + Robotiq 2F85 Pick and Place"
    )
    
    # Mode selection
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Run random action demo",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Run static robot demo (no movement)",
    )
    
    # Environment settings
    parser.add_argument(
        "--task-mode",
        type=str,
        default="pick_place",
        choices=["reach", "pick", "pick_place"],
        help="Task mode (default: pick_place)",
    )
    parser.add_argument(
        "--easy-mode",
        action="store_true",
        help="Use easier settings",
    )
    parser.add_argument(
        "--static-cube",
        action="store_true",
        help="Keep cube at fixed position",
    )
    
    # Evaluation settings
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (default: False)",
    )
    parser.add_argument(
        "--slow-motion",
        action="store_true",
        help="Run in slow motion",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.static:
        demo_static(args)
    elif args.random:
        demo_random(args)
    elif args.model_path:
        evaluate_model(args)
    else:
        print("Error: Must specify --model-path, --random, or --static")
        print("Run with --help for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()
