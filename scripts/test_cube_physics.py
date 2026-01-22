#!/usr/bin/env python3
"""
Test script to verify cube physics stability when gripper contacts cube.

Uses teleoperation-style slow movements to avoid robot instability.
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_cube_physics():
    """Test cube stability during gripper contact."""
    model_path = project_root / "assets" / "ur5e_robotiq_pick_place.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Get IDs
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    gripper_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
    pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")

    # Reset
    mujoco.mj_resetData(model, data)

    # Home position
    home_qpos = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

    # Pre-grasp position (above cube)
    pregrasp_qpos = np.array([0, -1.3, 1.9, -2.17, -1.57, 0])

    # Grasp position (lower to contact cube)
    grasp_qpos = np.array([0, -1.15, 2.0, -2.42, -1.57, 0])

    # Lift position
    lift_qpos = np.array([0, -1.4, 1.8, -1.97, -1.57, 0])

    # Initialize at home
    for i, jid in enumerate(joint_ids):
        data.qpos[model.jnt_qposadr[jid]] = home_qpos[i]

    # Set cube position
    cube_qpos_addr = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [0.32, 0.0, 0.445]
    data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1, 0, 0, 0]

    # Set controls to home
    data.ctrl[:6] = home_qpos
    data.ctrl[6] = 0  # Gripper open

    mujoco.mj_forward(model, data)

    print("=" * 60)
    print("CUBE PHYSICS TEST (Slow Motion)")
    print("=" * 60)

    initial_cube_pos = data.xpos[cube_body_id].copy()
    print(f"Initial cube position: {initial_cube_pos}")
    print(f"Initial EE position: {data.site_xpos[pinch_site_id]}")

    cube_flew = False
    max_cube_velocity = 0.0

    # Current control target
    current_ctrl = home_qpos.copy()
    gripper_ctrl = 0

    def smooth_move_to(target_qpos, target_gripper, steps, description):
        """Smoothly interpolate to target position.

        Note: gripper control range is now 0-0.8 (matches joint range)
        0 = open, 0.8 = closed
        """
        nonlocal current_ctrl, gripper_ctrl, cube_flew, max_cube_velocity

        print(f"\n{description}")
        start_qpos = current_ctrl.copy()
        start_gripper = gripper_ctrl

        for step in range(steps):
            # Smooth interpolation (ease in-out)
            t = step / steps
            t_smooth = t * t * (3 - 2 * t)  # Smoothstep

            # Interpolate joint targets
            current_ctrl = start_qpos + t_smooth * (target_qpos - start_qpos)
            gripper_ctrl = start_gripper + t_smooth * (target_gripper - start_gripper)

            data.ctrl[:6] = current_ctrl
            data.ctrl[6] = gripper_ctrl  # Now 0-0.8 range

            # Step simulation
            mujoco.mj_step(model, data)

            # Monitor cube
            cube_pos = data.xpos[cube_body_id].copy()
            cube_vel = data.qvel[model.jnt_dofadr[cube_joint_id]:model.jnt_dofadr[cube_joint_id]+3]
            vel_magnitude = np.linalg.norm(cube_vel)
            max_cube_velocity = max(max_cube_velocity, vel_magnitude)

            # Check if cube flew away
            if cube_pos[2] > 0.8 or cube_pos[2] < 0.3:
                cube_flew = True
                print(f"  ❌ CUBE FLEW AWAY at step {step}! Height: {cube_pos[2]:.3f}")
                return False

            if np.linalg.norm(cube_pos[:2] - initial_cube_pos[:2]) > 0.2:
                cube_flew = True
                print(f"  ❌ CUBE FLEW AWAY HORIZONTALLY at step {step}!")
                return False

            # Print status
            if step % 100 == 0 or step == steps - 1:
                ee_pos = data.site_xpos[pinch_site_id]
                dist = np.linalg.norm(ee_pos - cube_pos)
                grip_pos = data.qpos[model.jnt_qposadr[gripper_joint_id]]
                print(f"  Step {step:4d}: Dist={dist:.4f}m, Cube Z={cube_pos[2]:.4f}, "
                      f"Gripper={grip_pos:.3f}, Vel={vel_magnitude:.4f}")

        return True

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nStarting test sequence with slow movements...")

        # Phase 1: Move to pre-grasp position (above cube)
        # Gripper: 0 = open, 0.8 = closed
        if not smooth_move_to(pregrasp_qpos, 0, 500, "Phase 1: Moving to pre-grasp position (gripper open)"):
            pass  # Continue to show results
        else:
            viewer.sync()
            time.sleep(0.5)

            # Phase 2: Lower to grasp position (contact cube)
            if not smooth_move_to(grasp_qpos, 0, 500, "Phase 2: Lowering to contact cube (gripper open)"):
                pass
            else:
                viewer.sync()
                time.sleep(0.5)

                # Phase 3: Close gripper (0.8 = fully closed)
                if not smooth_move_to(grasp_qpos, 0.8, 500, "Phase 3: Closing gripper slowly"):
                    pass
                else:
                    viewer.sync()
                    time.sleep(0.5)

                    # Phase 4: Lift
                    if not smooth_move_to(lift_qpos, 0.8, 500, "Phase 4: Lifting cube"):
                        pass
                    else:
                        viewer.sync()
                        time.sleep(0.5)

        # Final report
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)

        final_cube_pos = data.xpos[cube_body_id].copy()
        ee_pos = data.site_xpos[pinch_site_id]
        gripper_pos = data.qpos[model.jnt_qposadr[gripper_joint_id]]
        dist = np.linalg.norm(ee_pos - final_cube_pos)

        print(f"Final cube position: {final_cube_pos}")
        print(f"Final EE position: {ee_pos}")
        print(f"Final distance EE-Cube: {dist:.4f}m")
        print(f"Gripper position: {gripper_pos:.3f} (0=open, 0.8=closed)")
        print(f"Max cube velocity: {max_cube_velocity:.4f} m/s")
        print(f"Cube height change: {final_cube_pos[2] - initial_cube_pos[2]:.4f}m")

        if cube_flew:
            print("\n❌ FAILED: Cube flew away during test!")
            print("   → Contact parameters need adjustment")
        elif final_cube_pos[2] > initial_cube_pos[2] + 0.03 and gripper_pos > 0.3:
            print("\n✅ SUCCESS: Cube was grasped and lifted!")
            print(f"   Lifted by {(final_cube_pos[2] - initial_cube_pos[2])*100:.1f}cm")
        elif dist < 0.06 and gripper_pos > 0.3:
            print("\n⚠️ PARTIAL: Cube contacted but not lifted properly")
        else:
            print("\n❌ FAILED: Cube not properly grasped")
            print(f"   Distance: {dist:.4f}m (need < 0.06m)")
            print(f"   Gripper: {gripper_pos:.3f} (need > 0.3)")

        print("=" * 60)

        # Keep viewer open
        print("\nViewer open - press ESC to close...")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    test_cube_physics()
