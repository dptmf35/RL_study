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
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.ur5e_robotiq_pick_place_env import UR5eRobotiqPickPlaceEnv
from envs.ur5e_robotiq_goal_env import UR5eRobotiqGoalEnv


def evaluate_model(args):
    """Evaluate a trained model."""
    print(f"Loading model from: {args.model_path}")

    # Detect model type
    use_goal_env = args.goal_env or "sac_her" in args.model_path.lower()

    # Create environment first (HER models need env for loading)
    if use_goal_env:
        print("Using SAC model with GoalEnv")
        base_env = UR5eRobotiqGoalEnv(
            render_mode="human" if not args.no_render else None,
            max_episode_steps=200,
            task_mode=args.task_mode,
            easy_mode=args.easy_mode,
        )
        env = DummyVecEnv([lambda: base_env])
        # HER models require env when loading
        model = SAC.load(args.model_path, env=env)
    else:
        print("Using PPO model with standard env")
        base_env = UR5eRobotiqPickPlaceEnv(
            render_mode="human" if not args.no_render else None,
            max_episode_steps=500,
            reward_type="dense",
            randomize_cube=not args.static_cube,
            randomize_target=False,
            task_mode=args.task_mode,
            easy_mode=args.easy_mode,
        )

        # Try to load VecNormalize stats
        vec_normalize_path = Path(args.model_path).parent.parent / "vec_normalize.pkl"
        if not vec_normalize_path.exists():
            vec_normalize_path = Path(args.model_path).parent / "vec_normalize.pkl"

        # Wrap with VecNormalize if available
        if vec_normalize_path.exists():
            print(f"Loading VecNormalize stats from: {vec_normalize_path}")
            env = DummyVecEnv([lambda: base_env])
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
        else:
            print("Warning: VecNormalize stats not found, using unnormalized observations")
            env = DummyVecEnv([lambda: base_env])

        model = PPO.load(args.model_path)
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    success_count = 0

    print(f"\nRunning {args.n_episodes} episodes...")
    print(f"Task mode: {args.task_mode}")
    print(f"Environment type: {'GoalEnv (SAC+HER)' if use_goal_env else 'Standard (PPO)'}")
    if args.task_mode == "reach":
        threshold = 0.05 if use_goal_env else 0.10
        print(f"Success threshold: {threshold}m\n")

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

            # Print real-time distance info every 10 steps
            if episode_length % 10 == 0 or done[0]:
                if use_goal_env:
                    # GoalEnv info format
                    dist = info[0].get('distance', 0)
                    success = info[0].get('is_success', False)
                    threshold = 0.05
                    status = "✓" if dist < threshold else " "
                    print(f"  Step {episode_length:3d} | Dist: {dist:.4f}m {status}| "
                          f"Threshold: {threshold}m | Success: {success} | "
                          f"Reward: {reward[0]:6.2f}")
                else:
                    # Standard env info format
                    dist = info[0].get('distance_to_cube', 0)
                    cube_h = info[0].get('cube_height', 0)
                    grasped = info[0].get('is_grasped', False)
                    success = info[0].get('task_success', False)

                    # Show reach threshold comparison
                    if args.task_mode == "reach":
                        threshold = 0.10
                        status = "✓" if dist < threshold else " "
                        print(f"  Step {episode_length:3d} | Dist: {dist:.4f}m {status}| "
                              f"Threshold: {threshold}m | Success: {success} | "
                              f"Reward: {reward[0]:6.2f}")
                    else:
                        print(f"  Step {episode_length:3d} | Dist to cube: {dist:.4f}m | "
                              f"Cube height: {cube_h:.3f}m | Grasped: {grasped} | "
                              f"Reward: {reward[0]:6.2f}")

            # Render (access base environment directly)
            if not args.no_render:
                base_env.render()

            # Slow motion if requested
            if args.slow_motion:
                time.sleep(0.05)

            # Check if done
            if done[0]:
                break

        # Log episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Get final distance from appropriate info key
        if use_goal_env:
            final_dist = info[0].get('distance', 0)
            is_success = info[0].get('is_success', False)
        else:
            final_dist = info[0].get('distance_to_cube', 0)
            is_success = info[0].get('task_success', False)

        episode_distances.append(final_dist)

        if is_success:
            success_count += 1
            status = "✅ SUCCESS"
        else:
            status = "❌ FAILED"

        # Show threshold comparison for reach task
        threshold_info = ""
        if args.task_mode == "reach":
            threshold = 0.05 if use_goal_env else 0.10
            threshold_info = f" (Threshold: {threshold:.3f}m)"

        print(f"\nEpisode {episode+1}/{args.n_episodes} {status}:")
        print(f"  Final distance: {final_dist:.4f}m{threshold_info}")
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length} steps")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary ({args.n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    # Show distance statistics for reach task
    if args.task_mode == "reach" and episode_distances:
        mean_dist = np.mean(episode_distances)
        min_dist = np.min(episode_distances)
        max_dist = np.max(episode_distances)
        print(f"Final distance: {mean_dist:.4f}m (min: {min_dist:.4f}m, max: {max_dist:.4f}m)")
        if use_goal_env:
            print(f"  → {sum(1 for d in episode_distances if d <= 0.05)} episodes within 5cm (GoalEnv threshold)")
            print(f"  → {sum(1 for d in episode_distances if d <= 0.03)} episodes within 3cm (very close)")
        else:
            print(f"  → {sum(1 for d in episode_distances if d <= 0.10)} episodes within 10cm threshold")
            print(f"  → {sum(1 for d in episode_distances if d <= 0.08)} episodes within 8cm (strict)")
            print(f"  → {sum(1 for d in episode_distances if d <= 0.05)} episodes within 5cm (very close)")

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
    parser.add_argument(
        "--goal-env",
        action="store_true",
        help="Use GoalEnv (for SAC+HER models)",
    )

    # Environment settings
    parser.add_argument(
        "--task-mode",
        type=str,
        default="reach",  # Changed default to reach for easier debugging
        choices=["reach", "pick", "pick_place"],
        help="Task mode (default: reach)",
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
