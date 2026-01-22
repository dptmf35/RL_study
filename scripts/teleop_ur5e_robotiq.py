#!/usr/bin/env python3
"""
Teleoperation Script for UR5e + Robotiq 2F85 Environment

Control the robot manually using keyboard to test the environment.
This helps verify that:
1. The gripper can physically reach and grasp the cube
2. Contact physics work correctly (cube doesn't fly away)
3. The robot kinematics allow the desired motions

Controls:
    Robot Arm (Joint-by-joint control):
        1/Q - Joint 0 (shoulder_pan) +/-
        2/W - Joint 1 (shoulder_lift) +/-
        3/E - Joint 2 (elbow) +/-
        4/R - Joint 3 (wrist_1) +/-
        5/T - Joint 4 (wrist_2) +/-
        6/Y - Joint 5 (wrist_3) +/-

    Gripper:
        O - Open gripper
        C - Close gripper

    Cartesian control (End Effector):
        Arrow Up/Down - Move EE forward/backward (X)
        Arrow Left/Right - Move EE left/right (Y)
        Page Up/Page Down - Move EE up/down (Z)

    Environment:
        Space - Reset environment
        P - Print current state
        H - Move to home position
        ESC - Quit
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TeleopController:
    def __init__(self):
        # Load model
        model_path = project_root / "assets" / "ur5e_robotiq_pick_place.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Get joint and actuator info
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                         for name in self.joint_names]

        # Gripper joint
        self.gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")

        # Sites for visualization
        self.pinch_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")

        # Control state
        self.joint_delta = 0.05  # Joint angle increment (radians)
        self.cartesian_delta = 0.01  # Cartesian position increment (meters)
        self.gripper_target = 0  # 0 = open, 255 = closed

        # Home position - arm extended towards table
        self.home_qpos = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

        # Key states for continuous control
        self.key_states = {}

        # Initialize
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        mujoco.mj_resetData(self.model, self.data)

        # Set robot to home position
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = self.home_qpos[i]

        # Open gripper
        gripper_qpos_addr = self.model.jnt_qposadr[self.gripper_joint_id]
        self.data.qpos[gripper_qpos_addr] = 0.0

        # Reset cube position (closer to robot)
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [0.32, 0.0, 0.445]
        self.data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1, 0, 0, 0]

        # Set initial controls
        self.data.ctrl[:6] = self.home_qpos
        self.data.ctrl[6] = 0  # Gripper open
        self.gripper_target = 0

        mujoco.mj_forward(self.model, self.data)
        print("Environment reset!")

    def move_to_home(self):
        """Smoothly move robot to home position."""
        self.data.ctrl[:6] = self.home_qpos
        print("Moving to home position...")

    def get_ee_position(self):
        """Get current end effector position."""
        return self.data.site_xpos[self.pinch_site_id].copy()

    def get_cube_position(self):
        """Get current cube position."""
        return self.data.xpos[self.cube_body_id].copy()

    def get_joint_positions(self):
        """Get current joint positions."""
        return np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                        for jid in self.joint_ids])

    def get_gripper_position(self):
        """Get current gripper joint position (0=open, ~0.8=closed)."""
        return self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]

    def print_state(self):
        """Print current state information."""
        ee_pos = self.get_ee_position()
        cube_pos = self.get_cube_position()
        joint_pos = self.get_joint_positions()
        gripper_pos = self.get_gripper_position()
        dist = np.linalg.norm(ee_pos - cube_pos)

        print("\n" + "="*60)
        print("CURRENT STATE")
        print("="*60)
        print(f"End Effector: x={ee_pos[0]:.4f}, y={ee_pos[1]:.4f}, z={ee_pos[2]:.4f}")
        print(f"Cube:         x={cube_pos[0]:.4f}, y={cube_pos[1]:.4f}, z={cube_pos[2]:.4f}")
        print(f"Distance EE->Cube: {dist:.4f} m")
        print(f"Gripper position: {gripper_pos:.3f} (0=open, 0.8=closed)")
        print(f"Joint positions (rad): {joint_pos}")
        print(f"Joint positions (deg): {np.degrees(joint_pos)}")
        print("="*60 + "\n")

    def apply_joint_delta(self, joint_idx, delta):
        """Apply delta to a specific joint."""
        current = self.data.ctrl[joint_idx]
        jid = self.joint_ids[joint_idx]
        jnt_range = self.model.jnt_range[jid]
        new_value = np.clip(current + delta, jnt_range[0], jnt_range[1])
        self.data.ctrl[joint_idx] = new_value

    def set_gripper(self, value):
        """Set gripper to specific value (0=open, 0.8=closed)."""
        self.gripper_target = np.clip(value, 0, 0.8)
        self.data.ctrl[6] = self.gripper_target

    def key_callback(self, key):
        """Handle key press events."""
        # Joint controls (positive direction)
        if key == ord('1'):
            self.apply_joint_delta(0, self.joint_delta)
        elif key == ord('2'):
            self.apply_joint_delta(1, self.joint_delta)
        elif key == ord('3'):
            self.apply_joint_delta(2, self.joint_delta)
        elif key == ord('4'):
            self.apply_joint_delta(3, self.joint_delta)
        elif key == ord('5'):
            self.apply_joint_delta(4, self.joint_delta)
        elif key == ord('6'):
            self.apply_joint_delta(5, self.joint_delta)

        # Joint controls (negative direction)
        elif key == ord('q') or key == ord('Q'):
            self.apply_joint_delta(0, -self.joint_delta)
        elif key == ord('w') or key == ord('W'):
            self.apply_joint_delta(1, -self.joint_delta)
        elif key == ord('e') or key == ord('E'):
            self.apply_joint_delta(2, -self.joint_delta)
        elif key == ord('r') or key == ord('R'):
            self.apply_joint_delta(3, -self.joint_delta)
        elif key == ord('t') or key == ord('T'):
            self.apply_joint_delta(4, -self.joint_delta)
        elif key == ord('y') or key == ord('Y'):
            self.apply_joint_delta(5, -self.joint_delta)

        # Gripper controls (0=open, 0.8=closed)
        elif key == ord('o') or key == ord('O'):
            self.set_gripper(0)  # Open
            print("Gripper: OPEN (0)")
        elif key == ord('c') or key == ord('C'):
            self.set_gripper(0.8)  # Close
            print("Gripper: CLOSE (0.8)")

        # Environment controls
        elif key == ord(' '):  # Space
            self.reset()
        elif key == ord('p') or key == ord('P'):
            self.print_state()
        elif key == ord('h') or key == ord('H'):
            self.move_to_home()

    def run(self):
        """Main loop with MuJoCo viewer."""
        print(__doc__)  # Print controls

        with mujoco.viewer.launch_passive(self.model, self.data,
                                          key_callback=self.key_callback) as viewer:
            print("\nViewer started! Use keyboard controls to operate the robot.")
            print("Press 'P' to print current state, Space to reset, ESC to quit.\n")

            while viewer.is_running():
                # Step simulation
                mujoco.mj_step(self.model, self.data)

                # Sync viewer
                viewer.sync()

                # Control rate
                time.sleep(0.002)  # ~500Hz


def main():
    print("="*60)
    print("UR5e + Robotiq 2F85 Teleoperation")
    print("="*60)

    controller = TeleopController()
    controller.run()


if __name__ == "__main__":
    main()
