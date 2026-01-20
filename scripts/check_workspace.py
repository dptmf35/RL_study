#!/usr/bin/env python3
"""
Workspace Check Script for UR5e Pick and Place Environment

This script helps verify:
1. Where the cube spawns
2. Where the robot's end effector can reach
3. If the cube positions are within the robot's workspace
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_workspace():
    """Check the robot workspace and cube spawn positions."""

    # Load model
    model_path = project_root / "assets" / "ur5e_pick_place.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Get IDs
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    # Joint names and IDs
    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]

    # Home position
    home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

    print("=" * 60)
    print("UR5e WORKSPACE ANALYSIS")
    print("=" * 60)

    # 1. Check home position end effector location
    mujoco.mj_resetData(model, data)
    for i, jid in enumerate(joint_ids):
        data.qpos[model.jnt_qposadr[jid]] = home_qpos[i]
    mujoco.mj_forward(model, data)

    home_ee_pos = data.site_xpos[ee_site_id].copy()
    print(f"\n1. Home Position End Effector Location:")
    print(f"   X: {home_ee_pos[0]:.3f}, Y: {home_ee_pos[1]:.3f}, Z: {home_ee_pos[2]:.3f}")

    # 2. Check cube spawn range
    cube_spawn_range = {
        "x": (0.3, 0.6),
        "y": (-0.3, 0.1),
        "z": 0.475
    }
    print(f"\n2. Cube Spawn Range:")
    print(f"   X: {cube_spawn_range['x']}")
    print(f"   Y: {cube_spawn_range['y']}")
    print(f"   Z: {cube_spawn_range['z']}")

    # 3. Robot base position
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_pos = data.xpos[base_body_id].copy()
    print(f"\n3. Robot Base Position:")
    print(f"   X: {base_pos[0]:.3f}, Y: {base_pos[1]:.3f}, Z: {base_pos[2]:.3f}")

    # 4. Sample random joint configurations and find reachable workspace
    print(f"\n4. Sampling Reachable Workspace (1000 random configs)...")

    reachable_positions = []
    for _ in range(1000):
        mujoco.mj_resetData(model, data)
        # Random joint positions within limits
        for i, jid in enumerate(joint_ids):
            jnt_range = model.jnt_range[jid]
            random_pos = np.random.uniform(jnt_range[0], jnt_range[1])
            data.qpos[model.jnt_qposadr[jid]] = random_pos
        mujoco.mj_forward(model, data)
        reachable_positions.append(data.site_xpos[ee_site_id].copy())

    reachable_positions = np.array(reachable_positions)

    print(f"   Reachable X range: [{reachable_positions[:,0].min():.3f}, {reachable_positions[:,0].max():.3f}]")
    print(f"   Reachable Y range: [{reachable_positions[:,1].min():.3f}, {reachable_positions[:,1].max():.3f}]")
    print(f"   Reachable Z range: [{reachable_positions[:,2].min():.3f}, {reachable_positions[:,2].max():.3f}]")

    # 5. Check if cube spawn positions are reachable
    print(f"\n5. Checking Cube Spawn Reachability...")

    # Test corner positions of spawn range
    test_positions = [
        (0.3, -0.3, 0.475),  # Near-left
        (0.3, 0.1, 0.475),   # Near-right
        (0.6, -0.3, 0.475),  # Far-left
        (0.6, 0.1, 0.475),   # Far-right
        (0.45, -0.1, 0.475), # Center
    ]

    for pos in test_positions:
        # Check distance from reachable positions
        distances = np.linalg.norm(reachable_positions - np.array(pos), axis=1)
        min_dist = distances.min()
        reachable = min_dist < 0.1  # Within 10cm

        status = "✓ REACHABLE" if reachable else "✗ POSSIBLY UNREACHABLE"
        print(f"   Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}): {status} (min dist: {min_dist:.3f}m)")

    # 6. Check target position
    target_pos = (0.6, 0.2, 0.475)
    distances = np.linalg.norm(reachable_positions - np.array(target_pos), axis=1)
    min_dist = distances.min()
    reachable = min_dist < 0.1
    status = "✓ REACHABLE" if reachable else "✗ POSSIBLY UNREACHABLE"
    print(f"\n6. Target Position ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}): {status}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)

    # Calculate recommended spawn range based on reachable workspace
    # Filter positions that are at table height and have good reach
    table_height = 0.475
    valid_positions = reachable_positions[
        (reachable_positions[:, 2] > table_height - 0.1) &
        (reachable_positions[:, 2] < table_height + 0.2)
    ]

    if len(valid_positions) > 0:
        recommended_x = (valid_positions[:, 0].min() + 0.05, valid_positions[:, 0].max() - 0.05)
        recommended_y = (valid_positions[:, 1].min() + 0.05, valid_positions[:, 1].max() - 0.05)

        print(f"Recommended cube spawn X range: ({max(0.2, recommended_x[0]):.2f}, {min(0.6, recommended_x[1]):.2f})")
        print(f"Recommended cube spawn Y range: ({max(-0.3, recommended_y[0]):.2f}, {min(0.3, recommended_y[1]):.2f})")

    return model, data


def visualize_workspace():
    """Open viewer to visually check the workspace."""
    model_path = project_root / "assets" / "ur5e_pick_place.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Reset to home position
    joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

    for i, jid in enumerate(joint_ids):
        data.qpos[model.jnt_qposadr[jid]] = home_qpos[i]

    mujoco.mj_forward(model, data)

    print("\nOpening MuJoCo viewer...")
    print("- Move the robot joints to check workspace")
    print("- Press ESC to close")

    viewer = mujoco.viewer.launch_passive(model, data)

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

    viewer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Open viewer for visual inspection")
    args = parser.parse_args()

    check_workspace()

    if args.visualize:
        visualize_workspace()
