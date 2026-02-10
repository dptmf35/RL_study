#!/usr/bin/env python3
"""Debug script to check end-effector orientation."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to euler angles (ZYX convention)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def main():
    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())

    print("="*60)
    print("END-EFFECTOR ORIENTATION DEBUG")
    print("="*60)

    obs, info = env.reset()

    # Get end-effector position
    ee_pos = env.data.site_xpos[env.ee_site_id]
    cube_pos = env.data.xpos[env.cube_body_id]

    # Get end-effector orientation (rotation matrix)
    ee_xmat = env.data.site_xmat[env.ee_site_id].reshape(3, 3)

    # Extract direction vectors
    x_axis = ee_xmat[:, 0]  # gripper X axis (finger direction)
    y_axis = ee_xmat[:, 1]  # gripper Y axis (open/close direction)
    z_axis = ee_xmat[:, 2]  # gripper Z axis (approach direction)

    # Desired direction: towards cube
    to_cube = cube_pos - ee_pos
    to_cube_norm = to_cube / np.linalg.norm(to_cube)

    # Check alignment
    z_alignment = np.dot(z_axis, to_cube_norm)
    downward_alignment = np.dot(z_axis, [0, 0, -1])

    print(f"\nEnd-effector position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
    print(f"Cube position: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
    print(f"\nOrientation vectors:")
    print(f"  X-axis (fingers): ({x_axis[0]:6.3f}, {x_axis[1]:6.3f}, {x_axis[2]:6.3f})")
    print(f"  Y-axis (open/close): ({y_axis[0]:6.3f}, {y_axis[1]:6.3f}, {y_axis[2]:6.3f})")
    print(f"  Z-axis (approach): ({z_axis[0]:6.3f}, {z_axis[1]:6.3f}, {z_axis[2]:6.3f})")

    print(f"\nDirection to cube: ({to_cube_norm[0]:6.3f}, {to_cube_norm[1]:6.3f}, {to_cube_norm[2]:6.3f})")
    print(f"Downward direction: ( 0.000,  0.000, -1.000)")

    print(f"\nAlignment metrics:")
    print(f"  Z-axis → cube: {z_alignment:6.3f} (1.0 = perfect alignment)")
    print(f"  Z-axis → down: {downward_alignment:6.3f} (1.0 = pointing straight down)")

    # Euler angles
    euler = rotation_matrix_to_euler(ee_xmat)
    print(f"\nEuler angles (rad): ({euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f})")
    print(f"Euler angles (deg): ({np.degrees(euler[0]):.1f}, {np.degrees(euler[1]):.1f}, {np.degrees(euler[2]):.1f})")

    # Current joint configuration
    joint_pos = np.array([env.data.qpos[env.model.jnt_qposadr[jid]]
                          for jid in env.arm_joint_ids])
    print(f"\nCurrent joint angles: {np.array2string(joint_pos, precision=3, suppress_small=True)}")

    print("\n" + "="*60)
    print("RECOMMENDATION:")
    if downward_alignment < 0.8:
        print("  ⚠️  Gripper not pointing sufficiently downward!")
        print("  → Adjust wrist joints (joint 4, 5) to orient gripper down")
    if abs(to_cube[0]) > 0.4:
        print(f"  ⚠️  Gripper too far from cube in X direction ({to_cube[0]:.3f}m)")
        print("  → Adjust shoulder/elbow to move gripper forward")
    if abs(z_alignment) < 0.5:
        print("  ℹ️  Gripper not aimed at cube (but may point down, which is OK)")

    print("="*60)
    env.close()

if __name__ == "__main__":
    main()
