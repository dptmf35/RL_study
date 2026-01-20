#!/usr/bin/env python3
"""
Hello Robot Stretch 2 Teleoperation Script

Keyboard controls for teleoperating the Stretch 2 robot in MuJoCo simulation.

Controls:
    Movement:
        W/S     - Move forward/backward
        A/D     - Turn left/right

    Arm:
        R/F     - Lift up/down
        T/G     - Extend/retract arm
        Y/H     - Wrist rotate left/right

    Gripper:
        O       - Open gripper
        P       - Close gripper

    Head:
        I/K     - Head tilt up/down
        J/L     - Head pan left/right

    Other:
        Q       - Quit
        Space   - Stop all movement
        ESC     - Quit
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import time

try:
    import glfw
except ImportError:
    print("Installing glfw...")
    import subprocess
    subprocess.check_call(["pip", "install", "glfw"])
    import glfw


class StretchTeleop:
    """Teleoperation controller for Hello Robot Stretch 2."""

    def __init__(self, model_path: str = None):
        """Initialize the teleoperation system."""
        if model_path is None:
            model_path = Path(__file__).parent / "assets" / "scene.xml"

        # Load model
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Actuator indices
        self.actuator_names = [
            "forward",      # 0: forward/backward
            "turn",         # 1: turn left/right
            "lift",         # 2: lift up/down
            "arm_extend",   # 3: arm extend/retract
            "wrist_yaw",    # 4: wrist rotation
            "grip",         # 5: gripper
            "head_pan",     # 6: head left/right
            "head_tilt",    # 7: head up/down
        ]

        # Get actuator IDs
        self.actuator_ids = {}
        for name in self.actuator_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self.actuator_ids[name] = aid

        print(f"Found actuators: {list(self.actuator_ids.keys())}")

        # Control values
        self.ctrl = np.zeros(self.model.nu)

        # Control speeds
        self.speeds = {
            "forward": 0.5,
            "turn": 0.3,
            "lift": 0.05,
            "arm_extend": 0.1,
            "wrist_yaw": 0.3,
            "grip": 0.01,
            "head_pan": 0.1,
            "head_tilt": 0.1,
        }

        # Key states
        self.key_states = {}

        # Viewer
        self.viewer = None

    def print_controls(self):
        """Print control instructions."""
        print("\n" + "="*60)
        print("STRETCH 2 TELEOPERATION CONTROLS")
        print("="*60)
        print("\nMovement:")
        print("  W/S     - Move forward/backward")
        print("  A/D     - Turn left/right")
        print("\nArm:")
        print("  R/F     - Lift up/down")
        print("  T/G     - Extend/retract arm")
        print("  Y/H     - Wrist rotate left/right")
        print("\nGripper:")
        print("  O       - Open gripper")
        print("  P       - Close gripper")
        print("\nHead:")
        print("  I/K     - Head tilt up/down")
        print("  J/L     - Head pan left/right")
        print("\nOther:")
        print("  Space   - Stop all movement")
        print("  Q/ESC   - Quit")
        print("="*60 + "\n")

    def key_callback(self, key):
        """Handle keyboard input."""
        # Movement
        if key == glfw.KEY_W:
            self._set_ctrl("forward", self.speeds["forward"])
        elif key == glfw.KEY_S:
            self._set_ctrl("forward", -self.speeds["forward"])
        elif key == glfw.KEY_A:
            self._set_ctrl("turn", self.speeds["turn"])
        elif key == glfw.KEY_D:
            self._set_ctrl("turn", -self.speeds["turn"])

        # Lift
        elif key == glfw.KEY_R:
            self._increment_ctrl("lift", self.speeds["lift"])
        elif key == glfw.KEY_F:
            self._increment_ctrl("lift", -self.speeds["lift"])

        # Arm extension
        elif key == glfw.KEY_T:
            self._increment_ctrl("arm_extend", self.speeds["arm_extend"])
        elif key == glfw.KEY_G:
            self._increment_ctrl("arm_extend", -self.speeds["arm_extend"])

        # Wrist
        elif key == glfw.KEY_Y:
            self._increment_ctrl("wrist_yaw", self.speeds["wrist_yaw"])
        elif key == glfw.KEY_H:
            self._increment_ctrl("wrist_yaw", -self.speeds["wrist_yaw"])

        # Gripper
        elif key == glfw.KEY_O:
            self._increment_ctrl("grip", -self.speeds["grip"])  # Open
        elif key == glfw.KEY_P:
            self._increment_ctrl("grip", self.speeds["grip"])   # Close

        # Head
        elif key == glfw.KEY_I:
            self._increment_ctrl("head_tilt", self.speeds["head_tilt"])
        elif key == glfw.KEY_K:
            self._increment_ctrl("head_tilt", -self.speeds["head_tilt"])
        elif key == glfw.KEY_J:
            self._increment_ctrl("head_pan", self.speeds["head_pan"])
        elif key == glfw.KEY_L:
            self._increment_ctrl("head_pan", -self.speeds["head_pan"])

        # Stop
        elif key == glfw.KEY_SPACE:
            self._stop_movement()

    def _set_ctrl(self, name: str, value: float):
        """Set control value directly."""
        if name in self.actuator_ids:
            idx = self.actuator_ids[name]
            self.ctrl[idx] = value

    def _increment_ctrl(self, name: str, delta: float):
        """Increment control value."""
        if name in self.actuator_ids:
            idx = self.actuator_ids[name]
            # Get control range
            ctrlrange = self.model.actuator_ctrlrange[idx]
            new_val = np.clip(self.ctrl[idx] + delta, ctrlrange[0], ctrlrange[1])
            self.ctrl[idx] = new_val

    def _stop_movement(self):
        """Stop wheel movement (keep arm/head positions)."""
        if "forward" in self.actuator_ids:
            self.ctrl[self.actuator_ids["forward"]] = 0
        if "turn" in self.actuator_ids:
            self.ctrl[self.actuator_ids["turn"]] = 0

    def run(self):
        """Run the teleoperation loop."""
        self.print_controls()

        # Create viewer with key callback
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self._viewer_key_callback,
        ) as viewer:
            self.viewer = viewer

            print("Simulation started. Press keys to control the robot.")
            print("Press Q or ESC to quit.\n")

            while viewer.is_running():
                # Apply controls
                self.data.ctrl[:] = self.ctrl

                # Step simulation
                mujoco.mj_step(self.model, self.data)

                # Decay wheel controls (for smoother stopping)
                if "forward" in self.actuator_ids:
                    self.ctrl[self.actuator_ids["forward"]] *= 0.95
                if "turn" in self.actuator_ids:
                    self.ctrl[self.actuator_ids["turn"]] *= 0.95

                # Sync viewer
                viewer.sync()

                # Small delay for stability
                time.sleep(0.001)

            print("\nTeleoperation ended.")

    def _viewer_key_callback(self, keycode):
        """Callback for viewer key events."""
        # Map mujoco viewer keycodes to glfw keycodes
        key_map = {
            ord('W'): glfw.KEY_W, ord('w'): glfw.KEY_W,
            ord('S'): glfw.KEY_S, ord('s'): glfw.KEY_S,
            ord('A'): glfw.KEY_A, ord('a'): glfw.KEY_A,
            ord('D'): glfw.KEY_D, ord('d'): glfw.KEY_D,
            ord('R'): glfw.KEY_R, ord('r'): glfw.KEY_R,
            ord('F'): glfw.KEY_F, ord('f'): glfw.KEY_F,
            ord('T'): glfw.KEY_T, ord('t'): glfw.KEY_T,
            ord('G'): glfw.KEY_G, ord('g'): glfw.KEY_G,
            ord('Y'): glfw.KEY_Y, ord('y'): glfw.KEY_Y,
            ord('H'): glfw.KEY_H, ord('h'): glfw.KEY_H,
            ord('O'): glfw.KEY_O, ord('o'): glfw.KEY_O,
            ord('P'): glfw.KEY_P, ord('p'): glfw.KEY_P,
            ord('I'): glfw.KEY_I, ord('i'): glfw.KEY_I,
            ord('K'): glfw.KEY_K, ord('k'): glfw.KEY_K,
            ord('J'): glfw.KEY_J, ord('j'): glfw.KEY_J,
            ord('L'): glfw.KEY_L, ord('l'): glfw.KEY_L,
            ord('Q'): glfw.KEY_Q, ord('q'): glfw.KEY_Q,
            ord(' '): glfw.KEY_SPACE,
            256: glfw.KEY_ESCAPE,  # ESC key
        }

        if keycode in key_map:
            key = key_map[keycode]
            if key in [glfw.KEY_Q, glfw.KEY_ESCAPE]:
                if self.viewer:
                    self.viewer.close()
            else:
                self.key_callback(key)


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Stretch 2 Teleoperation")
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Path to MuJoCo model XML")
    args = parser.parse_args()

    teleop = StretchTeleop(model_path=args.model)
    teleop.run()


if __name__ == "__main__":
    main()
