#!/usr/bin/env python3
"""Find home pose with horizontal arm and vertical gripper."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv

def evaluate_pose(env, joint_angles):
    """Evaluate how horizontal arm is and how vertical gripper is."""
    for i, jid in enumerate(env.arm_joint_ids):
        env.data.qpos[env.model.jnt_qposadr[jid]] = joint_angles[i]
    env.data.qpos[env.model.jnt_qposadr[env.gripper_joint_id]] = 0.0

    import mujoco
    mujoco.mj_forward(env.model, env.data)

    # Get positions
    base_pos = env.data.xpos[env.model.body("link_1").id]
    link2_pos = env.data.xpos[env.model.body("link_2").id]
    link3_pos = env.data.xpos[env.model.body("link_3").id]
    link4_pos = env.data.xpos[env.model.body("link_4").id]
    ee_pos = env.data.site_xpos[env.ee_site_id]

    # Arm horizontality: minimize Z variation
    z_variation = max(link2_pos[2], link3_pos[2], link4_pos[2]) - min(link2_pos[2], link3_pos[2], link4_pos[2])

    # Gripper verticality
    ee_xmat = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
    z_axis = ee_xmat[:, 2]
    downward = np.dot(z_axis, [0, 0, -1])

    # Arm horizontal extension ratio
    arm_vector = ee_pos - base_pos
    arm_horizontal = np.array([arm_vector[0], arm_vector[1], 0])
    arm_horizontal_length = np.linalg.norm(arm_horizontal)
    arm_total_length = np.linalg.norm(arm_vector)
    horizontal_ratio = arm_horizontal_length / arm_total_length if arm_total_length > 0 else 0

    # Position targets
    target_x = 0.65
    target_z = 0.88
    x_error = abs(ee_pos[0] - target_x)
    z_error = abs(ee_pos[2] - target_z)
    y_error = abs(ee_pos[1])

    # Score: prioritize horizontal arm + vertical gripper
    score = (
        z_variation * 10.0 +  # Heavily penalize arm tilt
        (1.0 - downward) * 15.0 +  # Heavily penalize non-vertical gripper
        (1.0 - horizontal_ratio) * 8.0 +  # Penalize non-horizontal extension
        x_error * 2.0 +
        y_error * 1.0 +
        z_error * 1.5
    )

    return {
        'score': score,
        'ee_pos': ee_pos.copy(),
        'z_variation': z_variation,
        'downward': downward,
        'horizontal_ratio': horizontal_ratio,
        'x_error': x_error,
        'z_error': z_error,
    }

def main():
    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())
    env.reset()

    print("="*75)
    print("SEARCHING FOR HORIZONTAL ARM + VERTICAL GRIPPER")
    print("="*75)
    print("Objectives:")
    print("  1. Arm horizontal (Z variation < 0.1m)")
    print("  2. Gripper vertical (downward > 0.95)")
    print("  3. Horizontal extension (ratio > 0.95)")
    print("  4. Good position (X≈0.65, Z≈0.88)")
    print()

    candidates = []

    # Focus on configurations that promote horizontal arm
    # Shoulder near 0 for horizontal, elbow to extend forward
    for j2 in np.linspace(-0.2, 0.2, 9):  # Shoulder: near horizontal
        for j3 in np.linspace(0.5, 1.5, 11):  # Elbow: moderate bend
            for j4 in np.linspace(-2.5, -1.0, 11):  # Wrist1: vary
                for j5 in np.linspace(-2.0, -1.0, 9):  # Wrist2: vertical gripper
                    config = np.array([0.0, j2, j3, j4, j5, 0.0])
                    metrics = evaluate_pose(env, config)

                    # Keep all configurations for analysis
                    candidates.append((config, metrics))

    if not candidates:
        print("❌ No good configurations found!")
        env.close()
        return

    candidates.sort(key=lambda x: x[1]['score'])

    print(f"Found {len(candidates)} candidates\n")
    print(f"Top 20:\n")
    print(f"{'#':<3} {'Score':<7} {'ZVar':<6} {'Down':<6} {'HRatio':<7} {'X':<7} {'Z':<7} {'Joints'}")
    print("-" * 75)

    for i, (config, m) in enumerate(candidates[:20]):
        ee = m['ee_pos']
        joints_str = f"[{config[0]:.2f},{config[1]:5.2f},{config[2]:5.2f},{config[3]:5.2f},{config[4]:5.2f},{config[5]:5.2f}]"
        print(f"{i+1:<3} {m['score']:<7.3f} {m['z_variation']:<6.3f} {m['downward']:<6.3f} {m['horizontal_ratio']:<7.3f} {ee[0]:<7.3f} {ee[2]:<7.3f} {joints_str}")

    best_config, best_metrics = candidates[0]

    print("\n" + "="*75)
    print("BEST CONFIGURATION:")
    print("="*75)
    print(f"Score: {best_metrics['score']:.3f}")
    print(f"\nArm horizontality:")
    print(f"  Z variation: {best_metrics['z_variation']:.3f}m (target < 0.1m)")
    print(f"  Horizontal ratio: {best_metrics['horizontal_ratio']:.3f} (target > 0.95)")
    print(f"\nGripper orientation:")
    print(f"  Downward alignment: {best_metrics['downward']:.3f} (target > 0.95)")
    print(f"\nPosition:")
    print(f"  End-effector: ({best_metrics['ee_pos'][0]:.3f}, {best_metrics['ee_pos'][1]:.3f}, {best_metrics['ee_pos'][2]:.3f})")
    print(f"  X error: {best_metrics['x_error']:.3f}m, Z error: {best_metrics['z_error']:.3f}m")
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
