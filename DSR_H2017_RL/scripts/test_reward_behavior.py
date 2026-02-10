#!/usr/bin/env python3
"""Test reward function behavior to understand agent learning."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv


def test_reward_behavior():
    """Test what actions give positive rewards."""
    print("="*70)
    print("REWARD BEHAVIOR TEST")
    print("="*70)

    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())
    obs, info = env.reset()

    print("\n1. Initial State")
    print("-"*70)
    ee_pos = env.data.site_xpos[env.ee_site_id]
    cube_pos = env.data.xpos[env.cube_body_id]

    print(f"End-effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
    print(f"Cube:         ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
    print(f"Distance XY:  {info['distance_xy']:.3f}m")
    print(f"Height error: {info['height_error']:.3f}m")
    print(f"Target height: cube Z + 0.10 = {cube_pos[2] + 0.10:.3f}m")

    # Compute initial reward
    _, reward_comps = env._compute_reward()
    print(f"\nInitial reward components:")
    for key, val in reward_comps.items():
        print(f"  {key:20s}: {val:7.3f}")
    print(f"  {'TOTAL':20s}: {sum(reward_comps.values()):7.3f}")

    print("\n2. Test Action: Move DOWN (towards cube Z)")
    print("-"*70)
    env.reset()

    # Action to move down (negative on shoulder/elbow)
    action_down = np.array([0, -0.5, 0, 0, 0, 0, -1.0])  # Close gripper, move down
    obs, reward, term, trunc, info = env.step(action_down)

    ee_pos = env.data.site_xpos[env.ee_site_id]
    print(f"After move down:")
    print(f"  End-effector Z: {ee_pos[2]:.3f}m")
    print(f"  Distance XY: {info['distance_xy']:.3f}m")
    print(f"  Height error: {info['height_error']:.3f}m")
    print(f"  Reward: {reward:.3f}")

    print("\n3. Test Action: Move TOWARDS CUBE XY")
    print("-"*70)
    obs, info = env.reset()

    ee_pos = env.data.site_xpos[env.ee_site_id]
    cube_pos = env.data.xpos[env.cube_body_id]

    # Calculate direction to cube in XY plane
    delta_xy = cube_pos[:2] - ee_pos[:2]

    print(f"Delta to cube: X={delta_xy[0]:.3f}, Y={delta_xy[1]:.3f}")

    # Action to move towards cube (adjust base rotation)
    # Positive base rotation if Y delta is negative
    base_action = -np.sign(delta_xy[1]) * 0.5

    action_towards = np.array([base_action, 0, 0, 0, 0, 0, -1.0])
    obs, reward, term, trunc, info = env.step(action_towards)

    ee_pos_after = env.data.site_xpos[env.ee_site_id]
    print(f"After move towards cube XY:")
    print(f"  End-effector: ({ee_pos_after[0]:.3f}, {ee_pos_after[1]:.3f}, {ee_pos_after[2]:.3f})")
    print(f"  Distance XY: {info['distance_xy']:.3f}m")
    print(f"  Height error: {info['height_error']:.3f}m")
    print(f"  Reward: {reward:.3f}")

    print("\n4. Compare Reward Changes")
    print("-"*70)

    # Reset and test multiple scenarios
    scenarios = []

    # Scenario A: Only move down
    env.reset()
    initial_reward = sum(env._compute_reward()[1].values())
    action = np.array([0, -0.3, 0, 0, 0, 0, -1.0])
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(action)
    final_reward_down = sum(env._compute_reward()[1].values())
    final_dist_down = info['distance_xy']
    scenarios.append(("Move DOWN only", initial_reward, final_reward_down, final_dist_down))

    # Scenario B: Move towards cube XY
    obs, info = env.reset()
    initial_reward = sum(env._compute_reward()[1].values())

    for _ in range(5):
        ee_pos = env.data.site_xpos[env.ee_site_id]
        cube_pos = env.data.xpos[env.cube_body_id]
        delta = cube_pos[:2] - ee_pos[:2]

        # Simple proportional control towards cube
        base_action = -np.sign(delta[1]) * 0.5
        action = np.array([base_action, 0, 0.2, 0, 0, 0, -1.0])
        obs, reward, term, trunc, info = env.step(action)

    final_reward_xy = sum(env._compute_reward()[1].values())
    final_dist_xy = info['distance_xy']
    scenarios.append(("Move TOWARDS XY", initial_reward, final_reward_xy, final_dist_xy))

    print(f"{'Scenario':<20} {'Initial':<10} {'Final':<10} {'Change':<10} {'Dist XY'}")
    print("-"*70)
    for name, init_r, final_r, dist in scenarios:
        change = final_r - init_r
        print(f"{name:<20} {init_r:>9.2f} {final_r:>9.2f} {change:>9.2f} {dist:>7.3f}m")

    print("\n5. Reward Weight Analysis")
    print("-"*70)

    # Sample different distances
    print(f"{'XY Dist':<10} {'Height Err':<12} {'Dist Penalty':<15} {'Height Penalty':<15} {'Total'}")
    print("-"*70)

    for xy_dist in [0.3, 0.2, 0.1, 0.05, 0.02]:
        for height_err in [0.25, 0.15, 0.05, 0.01]:
            dist_penalty = -3.0 * xy_dist
            height_penalty = -2.0 * abs(height_err)
            total = dist_penalty + height_penalty - 0.01  # time penalty
            print(f"{xy_dist:<10.2f} {height_err:<12.2f} {dist_penalty:<15.2f} {height_penalty:<15.2f} {total:>7.2f}")

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    # Check reward weight balance
    typical_xy_dist = 0.15
    typical_height_err = 0.15

    xy_penalty = -3.0 * typical_xy_dist
    height_penalty = -2.0 * abs(typical_height_err)

    print(f"Typical initial state:")
    print(f"  XY distance: {typical_xy_dist:.2f}m → penalty {xy_penalty:.2f}")
    print(f"  Height error: {typical_height_err:.2f}m → penalty {height_penalty:.2f}")

    if abs(xy_penalty) < abs(height_penalty):
        print("\n⚠️  XY PENALTY TOO WEAK!")
        print("   Agent prefers minimizing height error over XY alignment")
        print("   → Increase distance weight or decrease height weight")
    else:
        print("\n✅ XY penalty is stronger - should prioritize horizontal movement")

    env.close()


if __name__ == "__main__":
    test_reward_behavior()
