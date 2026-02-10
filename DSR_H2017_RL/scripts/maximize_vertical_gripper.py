#!/usr/bin/env python3
"""Maximize gripper vertical orientation while maintaining good position."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv

def evaluate_pose(env, joint_angles):
    """Focus on vertical gripper + good position."""
    for i, jid in enumerate(env.arm_joint_ids):
        env.data.qpos[env.model.jnt_qposadr[jid]] = joint_angles[i]
    env.data.qpos[env.model.jnt_qposadr[env.gripper_joint_id]] = 0.0

    import mujoco
    mujoco.mj_forward(env.model, env.data)

    ee_pos = env.data.site_xpos[env.ee_site_id]
    ee_xmat = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
    z_axis = ee_xmat[:, 2]
    downward = np.dot(z_axis, [0, 0, -1])

    target_x, target_z = 0.65, 0.88
    x_error = abs(ee_pos[0] - target_x)
    z_error = abs(ee_pos[2] - target_z)
    y_error = abs(ee_pos[1])

    # Focus on vertical gripper + position
    score = (
        (1.0 - downward) * 20.0 +  # MAXIMUM weight on vertical
        x_error * 2.0 +
        y_error * 1.0 +
        z_error * 2.0
    )

    return {
        'score': score,
        'downward': downward,
        'ee_pos': ee_pos.copy(),
        'x_error': x_error,
        'z_error': z_error,
    }

def main():
    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())
    env.reset()

    print("="*75)
    print("MAXIMIZE GRIPPER VERTICAL ORIENTATION")
    print("="*75)
    print("Primary goal: Gripper as vertical as possible (downward → 1.0)")
    print("Secondary: Good position (X≈0.65, Z≈0.88)\n")

    candidates = []

    # Wide search for maximum downward alignment
    for j2 in np.linspace(-0.3, 0.3, 7):
        for j3 in np.linspace(0.8, 2.2, 15):
            for j4 in np.linspace(-2.5, -1.2, 14):
                for j5 in np.linspace(-2.0, -1.0, 11):
                    for j6 in [-0.3, 0.0, 0.3]:
                        config = np.array([0.0, j2, j3, j4, j5, j6])
                        metrics = evaluate_pose(env, config)

                        # Keep if downward > 0.75 AND Z within range
                        if metrics['downward'] > 0.75 and 0.80 < metrics['ee_pos'][2] < 0.95:
                            candidates.append((config, metrics))

    if not candidates:
        print("❌ No configurations with downward > 0.75 found!")
        env.close()
        return

    candidates.sort(key=lambda x: x[1]['score'])

    print(f"Found {len(candidates)} candidates with downward > 0.75\n")
    print(f"Top 25 (sorted by score):\n")
    print(f"{'#':<3} {'Down':<6} {'Score':<7} {'X':<7} {'Z':<7} {'Joints'}")
    print("-" * 75)

    for i, (config, m) in enumerate(candidates[:25]):
        ee = m['ee_pos']
        joints_str = f"[{config[0]:.2f},{config[1]:5.2f},{config[2]:5.2f},{config[3]:5.2f},{config[4]:5.2f},{config[5]:5.2f}]"
        print(f"{i+1:<3} {m['downward']:<6.3f} {m['score']:<7.3f} {ee[0]:<7.3f} {ee[2]:<7.3f} {joints_str}")

    best_config, best_metrics = candidates[0]

    print("\n" + "="*75)
    print("RECOMMENDED CONFIGURATION:")
    print("="*75)
    print(f"Gripper downward alignment: {best_metrics['downward']:.4f} ⭐")
    print(f"End-effector: ({best_metrics['ee_pos'][0]:.3f}, {best_metrics['ee_pos'][1]:.3f}, {best_metrics['ee_pos'][2]:.3f})")
    print(f"Position errors: X={best_metrics['x_error']:.3f}m, Z={best_metrics['z_error']:.3f}m")
    print()
    print("Python code:")
    print("self.home_qpos = np.array([")
    names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]
    for i, val in enumerate(best_config):
        print(f"    {val:7.4f},  # {names[i]}")
    print("])")
    print("="*75)

    env.close()

if __name__ == "__main__":
    main()
