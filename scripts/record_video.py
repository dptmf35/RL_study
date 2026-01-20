#!/usr/bin/env python3
"""
Video Recording Script for UR5e Pick and Place

Records evaluation episodes as MP4 videos.
"""

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.ur5e_pick_place_env import UR5ePickPlaceEnv


def record_video(args):
    """Record evaluation episodes as video."""
    print(f"Recording {args.n_episodes} episodes...")

    # Create environment with rgb_array rendering
    env = UR5ePickPlaceEnv(
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
        reward_type="dense",
        randomize_cube=True,
        randomize_target=False,
    )

    # Load model if provided
    model = None
    if args.model_path:
        # Wrap for VecNormalize compatibility
        vec_env = DummyVecEnv([lambda: env])

        vec_normalize_path = Path(args.model_path).parent / "vec_normalize.pkl"
        if vec_normalize_path.exists():
            vec_env = VecNormalize.load(str(vec_normalize_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        model = PPO.load(args.model_path, env=vec_env)
        print(f"Loaded model from: {args.model_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(args.n_episodes):
        frames = []
        obs, info = env.reset()
        episode_reward = 0

        print(f"Recording episode {episode + 1}/{args.n_episodes}...")

        for step in range(args.max_steps):
            # Render frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            # Get action
            if model is not None:
                # Normalize observation if using VecNormalize
                if hasattr(model, 'env') and isinstance(model.env, VecNormalize):
                    obs_normalized = model.env.normalize_obs(obs.reshape(1, -1))
                    action, _ = model.predict(obs_normalized, deterministic=True)
                    action = action[0]
                else:
                    action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                    action = action[0]
            else:
                action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        # Save video
        video_path = output_dir / f"episode_{episode + 1}.mp4"
        print(f"Saving video to: {video_path} ({len(frames)} frames)")

        imageio.mimsave(
            str(video_path),
            frames,
            fps=args.fps,
            quality=8,
        )

        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Steps={len(frames)}, Success={info.get('task_success', False)}")

    env.close()
    print(f"\nAll videos saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Record evaluation videos")

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (uses random actions if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos",
        help="Output directory for videos (default: videos)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Number of episodes to record (default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_video(args)
