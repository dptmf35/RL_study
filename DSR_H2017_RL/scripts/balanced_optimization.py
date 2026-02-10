#!/usr/bin/env python3
"""Find balanced configuration: good downward orientation + correct height."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv

def evaluate_pose(env, joint_angles, target_x=0.65, target_z=0.88):
    """Balanced evaluation: position + orientation."""
    for i, jid in enumerate(env.arm_joint_ids):
        env.data.qpos[env.model.jnt_qposadr[jid]] = joint_angles[i]
    env.data.qpos[env.model.jnt_qposadr[env.gripper_joint_id]] = 0.0

    import mujoco
    mujoco.mj_forward(env.model, env.data)

    ee_pos = env.data.site_xpos[env.ee_site_id]
    ee_xmat = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
    z_axis = ee_xmat[:, 2]

    downward_alignment = np.dot(z_axis, [0, 0, -1])
    x_error = abs(ee_pos[0] - target_x)
    z_error = abs(ee_pos[2] - target_z)
    y_error = abs(ee_pos[1])

    # Balanced: good position + good orientation
    # Require downward > 0.5 as hard constraint
    if downward_alignment < 0.5:
        score = 999.0  # Penalty
    else:
        score = x_error * 1.5 + y_error * 0.8 + z_error * 2.0 + (1.0 - downward_alignment) * 5.0

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

    print("="*75)
    print("BALANCED OPTIMIZATION: Position + Downward Orientation")
    print("="*75)
    print("Target: X≈0.65, Z≈0.88±0.05, downward>0.5 (preferably >0.6)")
    print()

    candidates = []

    # Extended search
    for j2 in np.linspace(-0.4, -0.1, 4):
        for j3 in np.linspace(1.9, 2.3, 5):
            for j4 in np.linspace(-2.2, -1.6, 7):
                for j5 in np.linspace(-1.8, -1.3, 6):
                    for j6 in [-0.2, 0.0, 0.2]:
                        config = np.array([0.0, j2, j3, j4, j5, j6])
                        metrics = evaluate_pose(env, config)

                        # Only keep if downward > 0.5
                        if metrics['downward'] >= 0.5:
                            candidates.append((config, metrics))

    if not candidates:
        print("❌ No configuration found with downward > 0.5!")
        env.close()
        return

    candidates.sort(key=lambda x: x[1]['score'])

    print(f"Found {len(candidates)} valid configurations (downward>0.5)\n")
    print(f"Top 20:\n")
    print(f"{'#':<3} {'Score':<7} {'Down':<6} {'X':<7} {'Y':<7} {'Z':<7} {'Joints':<50}")
    print("-" * 75)

    for i, (config, metrics) in enumerate(candidates[:20]):
        ee = metrics['ee_pos']
        down = metrics['downward']
        joints_str = f"[{config[0]:.2f},{config[1]:5.2f},{config[2]:5.2f},{config[3]:5.2f},{config[4]:5.2f},{config[5]:5.2f}]"
        print(f"{i+1:<3} {metrics['score']:<7.3f} {down:<6.3f} {ee[0]:<7.3f} {ee[1]:<7.3f} {ee[2]:<7.3f} {joints_str}")

    best_config, best_metrics = candidates[0]

    print("\n" + "="*75)
    print("RECOMMENDED CONFIGURATION:")
    print("="*75)
    print(f"End-effector: ({best_metrics['ee_pos'][0]:.3f}, {best_metrics['ee_pos'][1]:.3f}, {best_metrics['ee_pos'][2]:.3f})")
    print(f"Downward alignment: {best_metrics['downward']:.3f}")
    print(f"Position errors: X={best_metrics['x_error']:.3f}m, Y={best_metrics['y_error']:.3f}m, Z={best_metrics['z_error']:.3f}m")
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
