#!/usr/bin/env python3
"""Debug what the trained agent is actually doing."""

import sys
from pathlib import Path
import numpy as np
import types

# Stub tensorboard before importing stable_baselines3
class _NullSummaryWriter:
    def __init__(self, *args, **kwargs): pass
    def add_scalar(self, *args, **kwargs): pass
    def add_histogram(self, *args, **kwargs): pass
    def add_text(self, *args, **kwargs): pass
    def flush(self): pass
    def close(self): pass

dummy_tensorboard = types.ModuleType("torch.utils.tensorboard")
dummy_tensorboard.SummaryWriter = _NullSummaryWriter
dummy_tensorboard.writer = types.SimpleNamespace(SummaryWriter=_NullSummaryWriter)
sys.modules["torch.utils.tensorboard"] = dummy_tensorboard

import torch
torch.utils.tensorboard = dummy_tensorboard

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import argparse


def debug_agent(model_path: str, vec_normalize_path: str):
    """Debug trained agent behavior."""

    print("="*70)
    print("TRAINED AGENT DEBUG")
    print("="*70)

    # Load environment
    def make_env():
        return DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())

    env = DummyVecEnv([make_env])

    # Load normalization stats
    if Path(vec_normalize_path).exists():
        print(f"\nLoading VecNormalize from {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

    # Load model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Run one episode with detailed logging
    print("\n" + "="*70)
    print("EPISODE TRACE")
    print("="*70)

    obs = env.reset()
    done = False
    step = 0
    total_reward = 0

    # Track success criteria over time
    success_criteria_log = []

    print(f"\n{'Step':<6} {'Distance':<10} {'Height':<10} {'Gripper':<10} {'Success':<8} {'Reward':<10}")
    print("-"*70)

    while not done and step < 300:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_reward += reward[0]
        step += 1

        # Get detailed info from actual env
        actual_env = env.envs[0]

        ee_pos = actual_env.data.site_xpos[actual_env.ee_site_id]
        cube_pos = actual_env.data.xpos[actual_env.cube_body_id]
        gripper_pos = actual_env.data.qpos[actual_env.model.jnt_qposadr[actual_env.gripper_joint_id]]

        distance_xy = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        height_error = ee_pos[2] - cube_pos[2] - 0.10  # target is +0.10m above cube

        success_criteria = {
            'distance_xy': distance_xy,
            'height_error': height_error,
            'gripper_pos': gripper_pos,
            'meets_xy': distance_xy < 0.04,
            'meets_height': abs(height_error) <= 0.02,
            'meets_gripper': gripper_pos <= 0.2,
        }
        success_criteria_log.append(success_criteria)

        if step % 30 == 0 or step < 10:
            print(f"{step:<6} {distance_xy:<10.4f} {height_error:<10.4f} {gripper_pos:<10.4f} "
                  f"{'✅' if actual_env.task_success else '❌':<8} {reward[0]:<10.2f}")

    print("-"*70)
    print(f"Final step: {step}, Total reward: {total_reward:.2f}")

    # Analyze why success wasn't achieved
    print("\n" + "="*70)
    print("SUCCESS CRITERIA ANALYSIS")
    print("="*70)

    # Check last 10 steps
    last_steps = success_criteria_log[-10:]

    print("\nLast 10 steps:")
    print(f"{'Criterion':<20} {'Met?':<10} {'Values'}")
    print("-"*70)

    xy_met = [s['meets_xy'] for s in last_steps]
    height_met = [s['meets_height'] for s in last_steps]
    gripper_met = [s['meets_gripper'] for s in last_steps]

    avg_distance = np.mean([s['distance_xy'] for s in last_steps])
    avg_height_err = np.mean([s['height_error'] for s in last_steps])
    avg_gripper = np.mean([s['gripper_pos'] for s in last_steps])

    print(f"{'XY Distance < 4cm':<20} {sum(xy_met)}/10     Avg: {avg_distance:.4f}m")
    print(f"{'Height Error < 2cm':<20} {sum(height_met)}/10     Avg: {avg_height_err:+.4f}m")
    print(f"{'Gripper Open < 0.2':<20} {sum(gripper_met)}/10     Avg: {avg_gripper:.4f}")

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    if sum(xy_met) >= 8:
        print("✅ XY alignment: GOOD")
    else:
        print("❌ XY alignment: POOR")

    if sum(height_met) >= 8:
        print("✅ Height alignment: GOOD")
    else:
        print("❌ Height alignment: POOR - Agent not matching target height!")
        print(f"   Average height error: {avg_height_err:+.4f}m (target: ±0.02m)")
        if avg_height_err > 0.05:
            print("   → Agent is TOO HIGH")
        elif avg_height_err < -0.05:
            print("   → Agent is TOO LOW")

    if sum(gripper_met) >= 8:
        print("✅ Gripper state: GOOD")
    else:
        print("❌ Gripper state: POOR - Agent not opening gripper!")
        print(f"   Average gripper pos: {avg_gripper:.4f} (target: < 0.2)")

    print("\n" + "="*70)
    print("REWARD HACKING ANALYSIS")
    print("="*70)

    # Calculate what rewards agent is getting
    typical_reward_per_step = total_reward / step

    print(f"\nAverage reward per step: {typical_reward_per_step:.2f}")
    print(f"Expected breakdown (if distance ~0.001m):")
    print(f"  - Distance penalty: -5.0 × {avg_distance:.4f} = {-5.0 * avg_distance:.2f}")
    print(f"  - Coarse align bonus: +2.0 (if < 10cm)")
    print(f"  - Tight align bonus: +5.0 (if < 4cm)")
    print(f"  - Height penalty: -1.0 × {abs(avg_height_err):.4f} = {-1.0 * abs(avg_height_err):.2f}")
    print(f"  - Time penalty: -0.01")

    expected_per_step = -5.0 * avg_distance + 2.0 + 5.0 - 1.0 * abs(avg_height_err) - 0.01
    print(f"\n  Expected per step: ~{expected_per_step:.2f}")
    print(f"  Actual per step: {typical_reward_per_step:.2f}")

    if typical_reward_per_step > 5.0 and not actual_env.task_success:
        print("\n⚠️  REWARD HACKING DETECTED!")
        print("   Agent exploits alignment bonuses without achieving success")
        print("   Recommendation: Reduce bonuses or make them conditional on full success")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--vec-normalize", type=str, default=None)
    args = parser.parse_args()

    # Auto-detect vec_normalize path
    if args.vec_normalize is None:
        model_dir = Path(args.model_path).parent.parent
        vec_normalize_path = model_dir / "vec_normalize.pkl"
    else:
        vec_normalize_path = Path(args.vec_normalize)

    debug_agent(args.model_path, str(vec_normalize_path))
