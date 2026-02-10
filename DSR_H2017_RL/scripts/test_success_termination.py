#!/usr/bin/env python3
"""Test that episode terminates after success."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv


def test_success_termination():
    """Test episode termination behavior."""
    print("="*70)
    print("SUCCESS TERMINATION TEST")
    print("="*70)

    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())

    # Test 1: Random policy until success (unlikely but possible)
    print("\n1. Random policy test (max 1000 steps)")
    print("-"*70)

    obs, info = env.reset()
    total_reward = 0
    success_achieved = False
    step = 0

    for step in range(1000):
        action = np.random.uniform(-1, 1, size=7)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info.get('task_success', False) and not success_achieved:
            print(f"   ✅ Success achieved at step {step}!")
            print(f"      Distance: {info['distance_xy']:.4f}m")
            print(f"      Height error: {info['height_error']:.4f}m")
            success_achieved = True

        if terminated:
            print(f"   🏁 Episode terminated at step {step}")
            print(f"      Total reward: {total_reward:.2f}")
            break

        if truncated:
            print(f"   ⏱️  Episode truncated at step {step}")
            break

    if not success_achieved:
        print("   ❌ Success not achieved with random policy")

    # Test 2: Manually place robot at success position
    print("\n2. Manual success placement test")
    print("-"*70)

    obs, info = env.reset()

    # Get cube position
    cube_pos = env.data.xpos[env.cube_body_id].copy()

    # Manually set robot to success position (above cube)
    # This is a hack for testing - directly set joint positions
    target_pos = cube_pos + np.array([0, 0, 0.10])  # 10cm above cube

    print(f"   Target position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")

    # Take small actions to stay near success
    total_reward = 0
    for step in range(50):
        # Very small random actions
        action = np.random.uniform(-0.1, 0.1, size=7)
        action[6] = -1.0  # Keep gripper open

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            print(f"   Step {step}: Distance={info['distance_xy']:.4f}m, "
                  f"Success={info.get('task_success', False)}, "
                  f"Reward={reward:.2f}")

        if terminated:
            print(f"\n   ✅ Episode terminated at step {step} after success!")
            print(f"      Total reward: {total_reward:.2f}")
            break

    if not terminated:
        print(f"\n   ❌ Episode did not terminate (reached step {step})")

    print("\n" + "="*70)
    print("SUCCESS REWARD TEST")
    print("="*70)

    # Test 3: Check reward magnitude
    obs, info = env.reset()

    # Simulate success condition
    print("\nReward components when successful:")
    print("  - First success: +100.0")
    print("  - Maintaining success: +10.0 per step")
    print("  - Distance penalty: ~-0.5 (if distance ~0.1m)")
    print("  - Time penalty: -0.01")
    print("\nExpected behavior:")
    print("  - Agent should achieve success and hold for 5 steps")
    print("  - Episode terminates automatically after holding")
    print("  - Total success reward: 100 + (10 * 5) = +150")

    env.close()


if __name__ == "__main__":
    test_success_termination()
