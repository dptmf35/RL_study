#!/usr/bin/env python3
"""Debug which joints actually move the robot towards cube."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv


def test_joint_effects():
    """Test what each joint does to end-effector position."""
    print("="*70)
    print("JOINT ACTION EFFECT TEST")
    print("="*70)

    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())
    obs, info = env.reset()

    ee_pos_init = env.data.site_xpos[env.ee_site_id].copy()
    cube_pos = env.data.xpos[env.cube_body_id].copy()

    print(f"\nInitial state:")
    print(f"  End-effector: ({ee_pos_init[0]:.3f}, {ee_pos_init[1]:.3f}, {ee_pos_init[2]:.3f})")
    print(f"  Cube:         ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
    print(f"  Delta needed: X={cube_pos[0]-ee_pos_init[0]:.3f}, Y={cube_pos[1]-ee_pos_init[1]:.3f}, Z={cube_pos[2]-ee_pos_init[2]:.3f}")

    print("\n" + "-"*70)
    print("Testing each joint (positive action = +1.0 for 10 steps)")
    print("-"*70)
    print(f"{'Joint':<10} {'ΔX':>8} {'ΔY':>8} {'ΔZ':>8} {'Effect Description'}")
    print("-"*70)

    joint_names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]

    for joint_idx in range(6):
        # Reset environment
        env.reset()
        obs, info = env.reset()
        ee_pos_before = env.data.site_xpos[env.ee_site_id].copy()

        # Apply positive action to this joint only
        for _ in range(10):
            action = np.zeros(7)
            action[joint_idx] = 1.0  # Maximum positive action
            action[6] = -1.0  # Keep gripper open
            obs, reward, term, trunc, info = env.step(action)

        ee_pos_after = env.data.site_xpos[env.ee_site_id].copy()
        delta = ee_pos_after - ee_pos_before

        # Determine primary effect
        abs_deltas = np.abs(delta)
        primary_axis = ['X', 'Y', 'Z'][np.argmax(abs_deltas)]
        primary_value = delta[np.argmax(abs_deltas)]

        effect_desc = f"Moves {primary_axis} {'+'if primary_value > 0 else ''}{primary_value:.3f}m"

        print(f"{joint_names[joint_idx]:<10} {delta[0]:>8.3f} {delta[1]:>8.3f} {delta[2]:>8.3f} {effect_desc}")

    print("\n" + "-"*70)
    print("Testing each joint (negative action = -1.0 for 10 steps)")
    print("-"*70)
    print(f"{'Joint':<10} {'ΔX':>8} {'ΔY':>8} {'ΔZ':>8} {'Effect Description'}")
    print("-"*70)

    for joint_idx in range(6):
        # Reset environment
        env.reset()
        obs, info = env.reset()
        ee_pos_before = env.data.site_xpos[env.ee_site_id].copy()

        # Apply negative action to this joint only
        for _ in range(10):
            action = np.zeros(7)
            action[joint_idx] = -1.0  # Maximum negative action
            action[6] = -1.0  # Keep gripper open
            obs, reward, term, trunc, info = env.step(action)

        ee_pos_after = env.data.site_xpos[env.ee_site_id].copy()
        delta = ee_pos_after - ee_pos_before

        # Determine primary effect
        abs_deltas = np.abs(delta)
        primary_axis = ['X', 'Y', 'Z'][np.argmax(abs_deltas)]
        primary_value = delta[np.argmax(abs_deltas)]

        effect_desc = f"Moves {primary_axis} {'+'if primary_value > 0 else ''}{primary_value:.3f}m"

        print(f"{joint_names[joint_idx]:<10} {delta[0]:>8.3f} {delta[1]:>8.3f} {delta[2]:>8.3f} {effect_desc}")

    print("\n" + "="*70)
    print("OPTIMAL ACTION STRATEGY")
    print("="*70)

    # Now test what actions move towards cube
    obs, info = env.reset()
    ee_pos = env.data.site_xpos[env.ee_site_id].copy()
    cube_pos = env.data.xpos[env.cube_body_id].copy()

    delta_to_cube = cube_pos - ee_pos

    print(f"\nTarget movement:")
    print(f"  Need to move X: {delta_to_cube[0]:+.3f}m")
    print(f"  Need to move Y: {delta_to_cube[1]:+.3f}m")
    print(f"  Need to move Z: {delta_to_cube[2]:+.3f}m (to 10cm above cube)")

    # Test combined action
    print("\n" + "-"*70)
    print("Testing combined action strategy")
    print("-"*70)

    strategies = [
        ("Base rotation only", np.array([1.0, 0, 0, 0, 0, 0, -1.0])),
        ("Shoulder+Elbow down", np.array([0, -0.5, -0.5, 0, 0, 0, -1.0])),
        ("Base+Shoulder+Elbow", np.array([0.5, -0.3, -0.3, 0, 0, 0, -1.0])),
        ("All joints balanced", np.array([0.3, -0.2, 0.2, -0.1, 0, 0, -1.0])),
    ]

    for strategy_name, action in strategies:
        env.reset()
        obs, info = env.reset()
        ee_before = env.data.site_xpos[env.ee_site_id].copy()
        cube_pos = env.data.xpos[env.cube_body_id].copy()
        dist_before = np.linalg.norm(ee_before[:2] - cube_pos[:2])

        for _ in range(10):
            obs, reward, term, trunc, info = env.step(action)

        ee_after = env.data.site_xpos[env.ee_site_id].copy()
        dist_after = np.linalg.norm(ee_after[:2] - cube_pos[:2])

        delta = ee_after - ee_before
        dist_change = dist_after - dist_before

        print(f"\n{strategy_name}:")
        print(f"  EE movement: X={delta[0]:+.3f}, Y={delta[1]:+.3f}, Z={delta[2]:+.3f}")
        print(f"  XY distance: {dist_before:.3f}m → {dist_after:.3f}m ({dist_change:+.3f}m)")
        if dist_change < 0:
            print(f"  ✅ Closer to cube!")
        else:
            print(f"  ❌ Further from cube")

    env.close()


if __name__ == "__main__":
    test_joint_effects()
