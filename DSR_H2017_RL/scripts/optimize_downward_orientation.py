#!/usr/bin/env python3
"""Optimize home pose specifically for maximum downward gripper orientation."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv

def evaluate_pose(env, joint_angles, target_x=0.65, target_z=0.88):
    """Evaluate pose with heavy weight on downward orientation."""
    # Set joints
    for i, jid in enumerate(env.arm_joint_ids):
        env.data.qpos[env.model.jnt_qposadr[jid]] = joint_angles[i]
    env.data.qpos[env.model.jnt_qposadr[env.gripper_joint_id]] = 0.0

    # Forward kinematics
    import mujoco
    mujoco.mj_forward(env.model, env.data)

    # Get metrics
    ee_pos = env.data.site_xpos[env.ee_site_id]
    ee_xmat = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
    z_axis = ee_xmat[:, 2]

    downward_alignment = np.dot(z_axis, [0, 0, -1])
    x_error = abs(ee_pos[0] - target_x)
    z_error = abs(ee_pos[2] - target_z)
    y_error = abs(ee_pos[1])

    # Heavy weight on downward orientation
    score = x_error * 1.0 + y_error * 0.5 + z_error * 1.0 + (1.0 - downward_alignment) * 10.0

    return {
        'score': score,
        'ee_pos': ee_pos.copy(),
        'downward': downward_alignment,
        'x_error': x_error,
        'z_error': z_error,
        'y_error': y_error,
    }

def main():
    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())
    env.reset()

    print("="*70)
    print("OPTIMIZING FOR MAXIMUM DOWNWARD ORIENTATION")
    print("="*70)
    print("Target: X≈0.65, Z≈0.88, gripper pointing STRAIGHT DOWN (alignment→1.0)")
    print()

    best_score = float('inf')
    best_config = None
    best_metrics = None

    candidates = []

    # Refined search around previous best
    # Keep shoulder/elbow close to previous best, vary wrists more
    for j2 in np.linspace(-0.5, -0.1, 5):
        for j3 in np.linspace(1.9, 2.3, 5):
            for j4 in np.linspace(-2.2, -1.5, 8):
                for j5 in np.linspace(-2.0, -1.2, 8):
                    for j6 in np.linspace(-0.3, 0.3, 5):
                        config = np.array([0.0, j2, j3, j4, j5, j6])
                        metrics = evaluate_pose(env, config)

                        candidates.append((config, metrics))

                        if metrics['score'] < best_score:
                            best_score = metrics['score']
                            best_config = config.copy()
                            best_metrics = metrics

    # Sort and show top 15
    candidates.sort(key=lambda x: x[1]['score'])

    print(f"\nTop 15 configurations (sorted by score):\n")
    print(f"{'#':<3} {'Score':<7} {'Down':<6} {'X':<7} {'Z':<7} {'Joints':<50}")
    print("-" * 70)

    for i, (config, metrics) in enumerate(candidates[:15]):
        ee = metrics['ee_pos']
        down = metrics['downward']
        joints_str = f"[{config[0]:.2f}, {config[1]:.2f}, {config[2]:.2f}, {config[3]:.2f}, {config[4]:.2f}, {config[5]:.2f}]"
        print(f"{i+1:<3} {metrics['score']:<7.3f} {down:<6.3f} {ee[0]:<7.3f} {ee[2]:<7.3f} {joints_str}")

    print("\n" + "="*70)
    print("BEST CONFIGURATION:")
    print("="*70)
    print(f"Score: {best_score:.3f}")
    print(f"End-effector: ({best_metrics['ee_pos'][0]:.3f}, {best_metrics['ee_pos'][1]:.3f}, {best_metrics['ee_pos'][2]:.3f})")
    print(f"Downward alignment: {best_metrics['downward']:.3f} ⬅️  MAIN METRIC")
    print(f"X error: {best_metrics['x_error']:.3f}m")
    print(f"Z error: {best_metrics['z_error']:.3f}m")
    print()
    print("Python code:")
    print(f"self.home_qpos = np.array([")
    names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]
    for i, val in enumerate(best_config):
        print(f"    {val:7.4f},  # {names[i]}")
    print("])")
    print("="*70)

    env.close()

if __name__ == "__main__":
    main()
