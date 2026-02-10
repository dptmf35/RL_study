#!/usr/bin/env python3
"""Search for optimal home pose that positions gripper near cube with good downward orientation."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv

def evaluate_pose(env, joint_angles, target_x=0.65, target_z=0.88):
    """Evaluate a home pose configuration."""
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

    # Score: lower is better
    score = x_error * 2.0 + y_error * 1.0 + z_error * 1.5 + (1.0 - downward_alignment) * 3.0

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
    print("SEARCHING FOR OPTIMAL HOME POSE")
    print("="*70)
    print("Target: X≈0.65, Y≈0, Z≈0.88, gripper pointing down")
    print()

    best_score = float('inf')
    best_config = None
    best_metrics = None

    # Grid search over key joints
    # Joint 2 (shoulder): -1.2 to -0.3
    # Joint 3 (elbow): 1.8 to 2.7
    # Joint 4 (wrist1): -2.0 to -1.0

    candidates = []

    for j2 in np.linspace(-1.2, -0.3, 7):
        for j3 in np.linspace(1.8, 2.7, 7):
            for j4 in np.linspace(-2.0, -1.0, 5):
                config = np.array([0.0, j2, j3, j4, -1.57, 0.0])
                metrics = evaluate_pose(env, config)

                candidates.append((config, metrics))

                if metrics['score'] < best_score:
                    best_score = metrics['score']
                    best_config = config.copy()
                    best_metrics = metrics

    # Sort and show top 10
    candidates.sort(key=lambda x: x[1]['score'])

    print("\nTop 10 configurations:\n")
    print(f"{'Rank':<5} {'Score':<8} {'X':<7} {'Y':<7} {'Z':<7} {'Down':<6} {'Joints':<40}")
    print("-" * 70)

    for i, (config, metrics) in enumerate(candidates[:10]):
        ee = metrics['ee_pos']
        down = metrics['downward']
        joints_str = f"[{config[0]:.2f}, {config[1]:.2f}, {config[2]:.2f}, {config[3]:.2f}, {config[4]:.2f}, {config[5]:.2f}]"
        print(f"{i+1:<5} {metrics['score']:<8.3f} {ee[0]:<7.3f} {ee[1]:<7.3f} {ee[2]:<7.3f} {down:<6.3f} {joints_str}")

    print("\n" + "="*70)
    print("BEST CONFIGURATION:")
    print("="*70)
    print(f"Score: {best_score:.3f}")
    print(f"End-effector: ({best_metrics['ee_pos'][0]:.3f}, {best_metrics['ee_pos'][1]:.3f}, {best_metrics['ee_pos'][2]:.3f})")
    print(f"Downward alignment: {best_metrics['downward']:.3f}")
    print(f"X error: {best_metrics['x_error']:.3f}m")
    print(f"Y error: {best_metrics['y_error']:.3f}m")
    print(f"Z error: {best_metrics['z_error']:.3f}m")
    print()
    print("Joints to use in code:")
    print(f"self.home_qpos = np.array([")
    for i, val in enumerate(best_config):
        names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]
        print(f"    {val:6.3f},  # {names[i]}")
    print("])")
    print("="*70)

    env.close()

if __name__ == "__main__":
    main()
