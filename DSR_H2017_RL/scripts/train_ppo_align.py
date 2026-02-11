#!/usr/bin/env python3
"""Train a PPO agent to align the DSR H2017 gripper above a cube."""

import argparse
import os
import sys
import types
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TENSORBOARD_NO_AWS_SDK", "1")

# Stub out tensorboard BEFORE importing stable_baselines3
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


# Create dummy tensorboard module before torch import
dummy_tensorboard = types.ModuleType("torch.utils.tensorboard")
dummy_tensorboard.SummaryWriter = _NullSummaryWriter
dummy_tensorboard.writer = types.SimpleNamespace(SummaryWriter=_NullSummaryWriter)
sys.modules["torch.utils.tensorboard"] = dummy_tensorboard

import torch
torch.utils.tensorboard = dummy_tensorboard  # type: ignore[attr-defined]

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv  # noqa: E402


def make_env(rank: int, seed: int, render_mode: str | None = None,
             randomize_home: bool = False, home_noise_scale: float = 0.15):
    """Create a thunk that builds a configured environment instance."""

    def _init():
        cfg = AlignmentConfig(
            randomize_home=randomize_home,
            home_noise_scale=home_noise_scale,
        )
        env = DSRH2017AlignEnv(render_mode=render_mode, config=cfg)
        env.reset(seed=seed + rank)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO agent for DSR H2017 alignment task"
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-dir", type=str, default=str(PROJECT_ROOT / "logs"))
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--randomize-home", action="store_true",
                        help="Randomize robot starting joint configuration each episode")
    parser.add_argument("--home-noise-scale", type=float, default=0.15,
                        help="Uniform noise range per joint in radians (default: 0.15, ~8.6 deg)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_dsr_align_{timestamp}"

    log_dir = Path(args.log_dir) / run_name
    model_dir = Path(args.model_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training run: {run_name}")
    print(f"Randomize home: {args.randomize_home} (noise={args.home_noise_scale:.2f} rad)")
    print(f"Logs:   {log_dir}")
    print(f"Models: {model_dir}")

    env_kwargs = dict(randomize_home=args.randomize_home,
                      home_noise_scale=args.home_noise_scale)

    if args.n_envs > 1:
        env = SubprocVecEnv(
            [make_env(i, args.seed, render_mode=None, **env_kwargs) for i in range(args.n_envs)]
        )
    else:
        env = DummyVecEnv([make_env(0, args.seed, render_mode=None, **env_kwargs)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval env: also randomize home to test generalization
    eval_env = DummyVecEnv([make_env(1000, args.seed + 1000, render_mode=None, **env_kwargs)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

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
        tensorboard_log=None,
        seed=args.seed,
        device=args.device,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // max(args.n_envs, 1), 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_dsr_align",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(str(model_dir / "final_model"))
    env.save(str(model_dir / "vec_normalize.pkl"))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
