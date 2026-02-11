#!/usr/bin/env python3
"""Evaluate SAC+HER policies for the DSR H2017 alignment task.

This script loads models trained with SAC+HER (GoalEnv, Dict obs space)
and runs evaluation episodes with optional visualisation.

Usage:
    python3 scripts/evaluate_sac_her.py \
        --model-path models/<run_name>/best/best_model.zip \
        --n-episodes 10

    python3 scripts/evaluate_sac_her.py --random --n-episodes 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs.dsr_h2017_goal_env import DSRH2017GoalEnv, GoalAlignConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SAC+HER for DSR H2017 alignment")
    p.add_argument("--model-path", type=Path, required=False)
    p.add_argument("--reward-type", type=str, default="sparse", choices=["sparse", "dense"])
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--render", action="store_true", default=True)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--slow-motion", action="store_true")
    p.add_argument("--random", action="store_true")
    p.add_argument("--randomize-home", action="store_true",
                   help="Randomize robot starting pose each episode")
    p.add_argument("--home-noise-scale", type=float, default=0.15,
                   help="Uniform noise per joint in radians (default: 0.15)")
    return p.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    render_mode = "human" if args.render else None
    cfg = GoalAlignConfig(
        randomize_home=args.randomize_home,
        home_noise_scale=args.home_noise_scale,
    )
    env = DSRH2017GoalEnv(render_mode=render_mode, reward_type=args.reward_type, config=cfg)

    model = None
    if args.model_path is not None:
        print(f"Loading SAC+HER model from {args.model_path}")
        model = SAC.load(str(args.model_path), env=env)

    success_count = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    final_distances: list[float] = []

    for episode in range(args.n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        step = 0

        for step in range(args.max_steps):
            if args.random or model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if args.render:
                env.render()
                if args.slow_motion:
                    time.sleep(0.02)

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        dist = info.get("distance", float("nan"))
        final_distances.append(dist)
        success = info.get("is_success", False)
        success_count += int(success)
        status = "SUCCESS" if success else "FAIL"
        print(
            f"Episode {episode + 1:3d}: {status} | "
            f"Reward={episode_reward:8.2f} | "
            f"Steps={step + 1:3d} | "
            f"Distance={dist:.4f}m"
        )

    print("\n=== SUMMARY ===")
    print(f"Episodes:      {args.n_episodes}")
    print(f"Success rate:  {success_count}/{args.n_episodes} ({100 * success_count / args.n_episodes:.0f}%)")
    print(f"Mean reward:   {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length:   {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Mean distance: {np.mean(final_distances):.4f}m +/- {np.std(final_distances):.4f}m")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    if args.no_render:
        args.render = False
    if not args.random and args.model_path is None:
        raise SystemExit("--model-path is required unless --random is specified")
    evaluate(args)
