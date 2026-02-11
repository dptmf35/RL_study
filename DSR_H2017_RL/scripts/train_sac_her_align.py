#!/usr/bin/env python3
"""Train a SAC + HER agent to align the DSR H2017 gripper above a cube.

SAC with Hindsight Experience Replay is well-suited for sparse-reward
manipulation tasks. The GoalEnv provides {observation, achieved_goal,
desired_goal} so HER can relabel failed trajectories with achieved goals.

Usage:
    python3 scripts/train_sac_her_align.py --total-timesteps 100000
    python3 scripts/train_sac_her_align.py --reward-type dense --total-timesteps 200000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Prevent pyOpenSSL crash (X509_V_FLAG_NOTIFY_POLICY AttributeError)
# by checking if tensorboard can be imported cleanly.
_tb_available = False
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    _tb_available = True
except Exception:
    pass

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from stable_baselines3 import HerReplayBuffer
    from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
except ImportError:
    from stable_baselines3.her import HerReplayBuffer, GoalSelectionStrategy

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs.dsr_h2017_goal_env import DSRH2017GoalEnv, GoalAlignConfig  # noqa: E402


def make_env(seed: int = 0, reward_type: str = "sparse",
             randomize_home: bool = False, home_noise_scale: float = 0.15):
    """Create a thunk that builds a GoalEnv instance."""
    def _init():
        cfg = GoalAlignConfig(
            randomize_home=randomize_home,
            home_noise_scale=home_noise_scale,
        )
        env = DSRH2017GoalEnv(render_mode=None, reward_type=reward_type, config=cfg)
        env.reset(seed=seed)
        return env
    return _init


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC+HER for DSR H2017 alignment")
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward-type", type=str, default="sparse",
                   choices=["sparse", "dense"])
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--n-sampled-goal", type=int, default=4)
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--save-freq", type=int, default=10_000)
    p.add_argument("--eval-freq", type=int, default=5_000)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--randomize-home", action="store_true",
                   help="Randomize robot starting joint configuration each episode")
    p.add_argument("--home-noise-scale", type=float, default=0.15,
                   help="Uniform noise range per joint in radians (default: 0.15, ~8.6 deg)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_her_dsr_align_{args.reward_type}_{timestamp}"
    log_dir = PROJECT_ROOT / args.log_dir / run_name
    model_dir = PROJECT_ROOT / args.model_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    tb_log = str(log_dir) if _tb_available else None

    print(f"Training run: {run_name}")
    print(f"Algorithm: SAC + HER (Hindsight Experience Replay)")
    print(f"Reward type: {args.reward_type}")
    print(f"Randomize home: {args.randomize_home} (noise={args.home_noise_scale:.2f} rad)")
    print(f"TensorBoard: {'enabled' if _tb_available else 'disabled (pyOpenSSL conflict)'}")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")

    env_kwargs = dict(randomize_home=args.randomize_home,
                      home_noise_scale=args.home_noise_scale)

    # HER requires a non-vectorised env for training
    env = make_env(args.seed, args.reward_type, **env_kwargs)()

    # Evaluation env (vectorised for EvalCallback) - also randomize to test generalization
    eval_env = DummyVecEnv([make_env(args.seed + 1000, args.reward_type, **env_kwargs)])

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=args.n_sampled_goal,
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
        ),
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=1000,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=tb_log,
        seed=args.seed,
        device=args.device,
    )

    print(f"\nModel: SAC with HER")
    print(f"Goal selection: FUTURE, n_sampled_goal={args.n_sampled_goal}")

    callbacks = [
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(model_dir),
            name_prefix="sac_her_dsr_align",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        ),
    ]

    print(f"\nStarting training for {args.total_timesteps} timesteps...")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted")

    model.save(str(model_dir / "final_model"))
    print(f"\nFinal model saved to: {model_dir / 'final_model'}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
