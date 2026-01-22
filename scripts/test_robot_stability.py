#!/usr/bin/env python3
"""
Simple test to verify robot arm stability without cube interaction.
Uses the teleop script's approach with keyboard control.
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_robot_stability():
    """Test robot stability with simple movements."""
    model_path = project_root / "assets" / "ur5e_robotiq_pick_place.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Get IDs
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
    pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

    # Reset
    mujoco.mj_resetData(model, data)

    # Home position
    home_qpos = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

    # Initialize
    for i, jid in enumerate(joint_ids):
        data.qpos[model.jnt_qposadr[jid]] = home_qpos[i]

    data.ctrl[:6] = home_qpos
    data.ctrl[6] = 0

    mujoco.mj_forward(model, data)

    print("=" * 60)
    print("ROBOT STABILITY TEST")
    print("=" * 60)
    print(f"Initial EE position: {data.site_xpos[pinch_site_id]}")
    print(f"Initial joint positions: {home_qpos}")
    print("\nTest: Hold position for 2 seconds...")

    # Test 1: Hold position stability
    with mujoco.viewer.launch_passive(model, data) as viewer:
        ee_positions = []

        for step in range(1000):
            data.ctrl[:6] = home_qpos
            data.ctrl[6] = 0
            mujoco.mj_step(model, data)

            ee_pos = data.site_xpos[pinch_site_id].copy()
            ee_positions.append(ee_pos)

            if step % 200 == 0:
                print(f"  Step {step}: EE = {ee_pos}")

            viewer.sync()
            time.sleep(0.002)

        ee_positions = np.array(ee_positions)
        ee_variance = np.var(ee_positions, axis=0)
        print(f"\nEE position variance: {ee_variance}")

        if np.max(ee_variance) < 0.0001:
            print("✅ Position holding: STABLE")
        else:
            print("❌ Position holding: UNSTABLE")

        # Test 2: Slow joint movement
        print("\nTest: Slow joint movement...")
        target_qpos = home_qpos.copy()

        for joint_idx in range(6):
            print(f"\n  Moving joint {joint_idx} ({joint_names[joint_idx]})...")

            # Move joint by 0.2 rad
            original_val = target_qpos[joint_idx]
            target_val = original_val + 0.2

            for step in range(200):
                t = step / 200
                target_qpos[joint_idx] = original_val + t * 0.2
                data.ctrl[:6] = target_qpos
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)

            ee_pos = data.site_xpos[pinch_site_id]
            actual_joint = data.qpos[model.jnt_qposadr[joint_ids[joint_idx]]]
            error = abs(actual_joint - target_val)

            print(f"    Target: {target_val:.3f}, Actual: {actual_joint:.3f}, Error: {error:.4f}")

            if error > 0.1:
                print(f"    ❌ Joint {joint_idx} tracking error too large!")
            else:
                print(f"    ✅ Joint {joint_idx} tracking OK")

            # Move back
            for step in range(200):
                t = step / 200
                target_qpos[joint_idx] = target_val - t * 0.2
                data.ctrl[:6] = target_qpos
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)

        # Test 3: Gripper
        # Control range: 0-0.8 (now matches joint range)
        print("\nTest: Gripper open/close...")
        print("  (Control: 0-0.8, Joint: 0-0.8)")
        for grip_target, expected in [(0, "open ~0"), (0.8, "closed ~0.8"), (0, "open ~0")]:
            print(f"  Gripper ctrl={grip_target} (expected: {expected})")
            for step in range(300):
                data.ctrl[6] = grip_target
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)

            grip_pos = data.qpos[model.jnt_qposadr[gripper_joint_id]]
            status = "✅" if (grip_target == 0 and grip_pos < 0.1) or (grip_target == 0.8 and grip_pos > 0.6) else "⚠️"
            print(f"    {status} Joint position: {grip_pos:.3f}")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print("Final EE position:", data.site_xpos[pinch_site_id])

        print("\nViewer open - press ESC to close...")
        while viewer.is_running():
            data.ctrl[:6] = home_qpos
            data.ctrl[6] = 0
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    test_robot_stability()
