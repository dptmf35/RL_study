#!/usr/bin/env python3
"""Check if arm is horizontal and gripper is vertical."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv

def main():
    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())
    env.reset()

    print("="*70)
    print("ARM HORIZONTAL & GRIPPER VERTICAL CHECK")
    print("="*70)

    # Get relevant body/site positions
    base_pos = env.data.xpos[env.model.body("link_1").id]
    link2_pos = env.data.xpos[env.model.body("link_2").id]
    link3_pos = env.data.xpos[env.model.body("link_3").id]
    link4_pos = env.data.xpos[env.model.body("link_4").id]
    ee_pos = env.data.site_xpos[env.ee_site_id]

    # Check arm horizontality
    # For horizontal arm, Z coordinates should be similar
    print("\nLink heights (Z coordinates):")
    print(f"  Link 1 (base):  Z = {base_pos[2]:.3f}")
    print(f"  Link 2:         Z = {link2_pos[2]:.3f}")
    print(f"  Link 3 (elbow): Z = {link3_pos[2]:.3f}")
    print(f"  Link 4:         Z = {link4_pos[2]:.3f}")
    print(f"  End-effector:   Z = {ee_pos[2]:.3f}")

    z_variation = max(link2_pos[2], link3_pos[2], link4_pos[2]) - min(link2_pos[2], link3_pos[2], link4_pos[2])
    print(f"\nZ variation in arm links: {z_variation:.3f}m")

    if z_variation < 0.1:
        print("  ✅ Arm is relatively horizontal")
    else:
        print(f"  ❌ Arm is tilted (variation {z_variation:.3f}m > 0.1m threshold)")

    # Check gripper orientation
    ee_xmat = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
    z_axis = ee_xmat[:, 2]  # Gripper approach direction

    downward = np.dot(z_axis, [0, 0, -1])
    print(f"\nGripper orientation:")
    print(f"  Z-axis direction: ({z_axis[0]:6.3f}, {z_axis[1]:6.3f}, {z_axis[2]:6.3f})")
    print(f"  Downward alignment: {downward:.3f}")

    if downward > 0.95:
        print("  ✅ Gripper is vertical (nearly perfect)")
    elif downward > 0.8:
        print("  ⚠️  Gripper is mostly vertical (good)")
    elif downward > 0.6:
        print("  ⚠️  Gripper is pointing down but tilted")
    else:
        print(f"  ❌ Gripper is not vertical enough ({downward:.3f} < 0.6)")

    # Check arm extension direction
    arm_vector = ee_pos - base_pos
    arm_horizontal = np.array([arm_vector[0], arm_vector[1], 0])
    arm_horizontal_length = np.linalg.norm(arm_horizontal)
    arm_total_length = np.linalg.norm(arm_vector)

    horizontal_ratio = arm_horizontal_length / arm_total_length if arm_total_length > 0 else 0

    print(f"\nArm extension:")
    print(f"  Horizontal reach: {arm_horizontal_length:.3f}m")
    print(f"  Total reach: {arm_total_length:.3f}m")
    print(f"  Horizontal ratio: {horizontal_ratio:.3f}")

    if horizontal_ratio > 0.95:
        print("  ✅ Arm extends horizontally")
    elif horizontal_ratio > 0.85:
        print("  ⚠️  Arm mostly horizontal")
    else:
        print(f"  ❌ Arm is too tilted ({horizontal_ratio:.3f} < 0.85)")

    # Current joint configuration
    joint_pos = np.array([env.data.qpos[env.model.jnt_qposadr[jid]]
                          for jid in env.arm_joint_ids])
    print(f"\nCurrent joint angles:")
    print(f"  {np.array2string(joint_pos, precision=3, suppress_small=True)}")

    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    issues = []
    if z_variation >= 0.1:
        issues.append(f"Arm tilted (Z variation {z_variation:.3f}m)")
    if downward < 0.8:
        issues.append(f"Gripper not vertical enough ({downward:.3f})")
    if horizontal_ratio < 0.85:
        issues.append(f"Arm not extending horizontally ({horizontal_ratio:.3f})")

    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration looks good!")
    print("="*70)

    env.close()

if __name__ == "__main__":
    main()
