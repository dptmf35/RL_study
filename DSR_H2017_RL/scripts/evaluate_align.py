#!/usr/bin/env python3
"""Evaluate or visualise policies for the DSR H2017 alignment task."""

import argparse
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TENSORBOARD_NO_AWS_SDK", "1")
class _NullSummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


dummy_tensorboard = types.ModuleType("torch.utils.tensorboard")
dummy_tensorboard.SummaryWriter = _NullSummaryWriter
dummy_tensorboard.writer = types.SimpleNamespace(SummaryWriter=_NullSummaryWriter)
sys.modules["torch.utils.tensorboard"] = dummy_tensorboard

import torch

torch.utils.tensorboard = dummy_tensorboard  # type: ignore[attr-defined]

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv  # noqa: E402


def build_env(render_mode: str | None) -> DummyVecEnv:
    return DummyVecEnv([
        lambda: DSRH2017AlignEnv(render_mode=render_mode, config=AlignmentConfig())
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DSR H2017 alignment policies"
    )
    parser.add_argument("--model-path", type=Path, required=False)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--slow-motion", action="store_true")
    parser.add_argument("--random", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    env = build_env("human" if args.render else None)

    vecnorm_path = None
    if args.model_path is not None:
        for level in range(3):
            candidate = args.model_path.parents[level] / "vec_normalize.pkl"
            if candidate.exists():
                vecnorm_path = candidate
                break

    if vecnorm_path is not None:
        print(f"Loading VecNormalize statistics from {vecnorm_path}")
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False

    model = None
    if args.model_path is not None:
        print(f"Loading model from {args.model_path}")
        model = PPO.load(str(args.model_path), env=env)

    success_count = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for episode in range(args.n_episodes):
        obs = env.reset()
        episode_reward = 0.0
        for step in range(args.max_steps):
            if args.random or model is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                if args.random:
                    action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)

            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

            if args.render:
                env.envs[0].render()
                if args.slow_motion:
                    time.sleep(0.02)

            if done[0]:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        final_info = info[0]
        success = final_info.get("task_success", False)
        success_count += int(success)
        status = "SUCCESS" if success else "FAIL"
        print(
            f"Episode {episode + 1}: {status} | Reward={episode_reward:.2f} | Steps={step + 1} | Distance={final_info.get('distance_xy', np.nan):.3f}"
        )

    print("\n=== SUMMARY ===")
    print(f"Episodes: {args.n_episodes}")
    print(f"Success rate: {success_count}/{args.n_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    if args.no_render:
        args.render = False
    if not args.random and args.model_path is None:
        raise SystemExit("--model-path is required unless --random is specified")
    evaluate(args)
