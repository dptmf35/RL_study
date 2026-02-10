#!/usr/bin/env python3
"""
Interactive Teleoperation for DSR H2017 Robot

Controls (키를 누르고 있으면 계속 움직임!):
  Joint controls:
    1/q - Joint 1 (base) decrease/increase
    2/w - Joint 2 (shoulder) decrease/increase
    3/e - Joint 3 (elbow) decrease/increase
    4/r - Joint 4 (wrist1) decrease/increase
    5/t - Joint 5 (wrist2) decrease/increase
    6/y - Joint 6 (wrist3) decrease/increase

  Gripper:
    g/h - Gripper close/open

  Step size:
    [/] - Decrease/increase step size

  Actions:
    SPACE - Reset to saved home position
    p - Print current joint angles (copy to code)
    s - Save current position as new home
    o - Save to environment file (auto-update)
    i - Show info (positions, distances)
    ESC - Exit

Requirements:
    pip install pynput
    (Usually works without sudo on Linux)
"""

import sys
from pathlib import Path
import numpy as np
import time
import re
from threading import Lock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv
import mujoco
import mujoco.viewer

try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("⚠️  'pynput' library not found. Install with: pip install pynput\n")


class TeleopController:
    def __init__(self):
        self.env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())
        self.env.reset()

        # Joint control parameters
        self.step_size = 0.01  # radians per frame (smaller = safer)
        self.gripper_step = 0.03

        # Current joint angles
        self.current_joints = self.env.home_qpos.copy()
        self.current_gripper = 0.0

        # Saved home position
        self.saved_home = self.env.home_qpos.copy()

        # Key press tracking
        self.keys_pressed = set()
        self.keys_lock = Lock()
        self.last_action_time = {}

        # Debounce for single-press actions (P, S, I, SPACE, ESC)
        self.action_cooldown = 0.3  # seconds

        # Keyboard listener
        self.listener = None

        print("="*70)
        print("DSR H2017 TELEOPERATION - 실시간 조작 모드")
        print("="*70)
        print("\n초기 home position:")
        self._print_joints()
        print("\n✨ 키를 누르고 있으면 로봇이 계속 움직입니다!")
        print("⚡ Self-collision 방지 기능 활성화됨")
        print("📏 Step size: [/] 키로 조절 가능")
        print("="*70)

    def _apply_state(self):
        """Apply current joint angles to simulation."""
        for i, jid in enumerate(self.env.arm_joint_ids):
            self.env.data.qpos[self.env.model.jnt_qposadr[jid]] = self.current_joints[i]

        self.env.data.qpos[self.env.model.jnt_qposadr[self.env.gripper_joint_id]] = self.current_gripper

        # Clear velocities to avoid instability
        self.env.data.qvel[:] = 0.0

        # Forward kinematics
        mujoco.mj_forward(self.env.model, self.env.data)

    def _check_collision(self):
        """Check for self-collisions in current configuration."""
        # Check for any active contacts
        for i in range(self.env.data.ncon):
            contact = self.env.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Get body IDs from geom IDs
            body1 = self.env.model.geom_bodyid[geom1]
            body2 = self.env.model.geom_bodyid[geom2]

            # Get body names
            body1_name = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_BODY, body2)

            # Check for self-collision (robot parts colliding with each other)
            if body1_name and body2_name:
                # Ignore floor/table collisions and gripper internal collisions
                if (body1_name.startswith('link') or body1_name == 'base_link') and \
                   (body2_name.startswith('link') or body2_name == 'base_link'):
                    # Self-collision detected!
                    return True

        return False

    def _print_joints(self):
        """Print current joint configuration."""
        print(f"\nJoint angles:")
        names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]
        for i, name in enumerate(names):
            deg = np.degrees(self.current_joints[i])
            print(f"  {name:8s} (J{i+1}): {self.current_joints[i]:7.4f} rad ({deg:6.1f}°)")
        print(f"  Gripper:  {self.current_gripper:.3f}")

    def _print_code(self):
        """Print code to copy into environment file."""
        print("\n" + "="*70)
        print("📋 COPY THIS TO YOUR CODE:")
        print("="*70)
        print("self.home_qpos = np.array([")
        names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]
        for i, name in enumerate(names):
            print(f"    {self.current_joints[i]:8.5f},  # {name}")
        print("])")
        print("="*70)

    def _print_info(self):
        """Print positions and distances."""
        ee_pos = self.env.data.site_xpos[self.env.ee_site_id]
        cube_pos = self.env.data.xpos[self.env.cube_body_id]

        # Orientation
        ee_xmat = self.env.data.site_xmat[self.env.ee_site_id].reshape(3, 3)
        z_axis = ee_xmat[:, 2]
        downward = np.dot(z_axis, [0, 0, -1])

        # Distances
        dist_xy = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        dist_3d = np.linalg.norm(ee_pos - cube_pos)

        print("\n" + "="*70)
        print("📊 CURRENT STATE INFO")
        print("="*70)
        print(f"End-effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
        print(f"Cube:         ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
        print(f"Distance XY:  {dist_xy:.3f}m")
        print(f"Distance 3D:  {dist_3d:.3f}m")
        print(f"Gripper Z-axis: ({z_axis[0]:6.3f}, {z_axis[1]:6.3f}, {z_axis[2]:6.3f})")
        print(f"Downward alignment: {downward:.3f} (1.0 = perfect vertical)")
        if downward > 0.8:
            print("  ✅ Gripper orientation: Excellent!")
        elif downward > 0.6:
            print("  ✓  Gripper orientation: Good")
        else:
            print("  ⚠️  Gripper orientation: Needs improvement")
        print("="*70)

    def adjust_joint(self, joint_idx, delta):
        """Adjust a joint by delta."""
        # Save previous state in case we need to revert
        prev_joints = self.current_joints.copy()

        self.current_joints[joint_idx] += delta

        # Clip to joint limits
        jid = self.env.arm_joint_ids[joint_idx]
        jrange = self.env.model.jnt_range[jid]
        if np.isfinite(jrange[0]) and np.isfinite(jrange[1]):
            self.current_joints[joint_idx] = np.clip(
                self.current_joints[joint_idx], jrange[0], jrange[1]
            )

        self._apply_state()

        # Check for collisions
        if self._check_collision():
            # Revert to previous state
            self.current_joints = prev_joints
            self._apply_state()

    def adjust_gripper(self, delta):
        """Adjust gripper opening."""
        self.current_gripper += delta
        self.current_gripper = np.clip(self.current_gripper, 0.0, 0.8)
        self._apply_state()

    def reset_to_home(self):
        """Reset to saved home position."""
        self.current_joints = self.saved_home.copy()
        self.current_gripper = 0.0
        self._apply_state()
        print("\n✅ Reset to saved home position")
        self._print_joints()

    def save_home(self):
        """Save current position as new home."""
        self.saved_home = self.current_joints.copy()
        print("\n💾 Saved current position as new home!")

    def save_to_env_file(self):
        """Save current position to environment file."""
        env_file = PROJECT_ROOT / "envs" / "dsr_h2017_align_env.py"

        try:
            # Read current file
            with open(env_file, 'r') as f:
                content = f.read()

            # Create new home_qpos array string
            new_qpos = "self.home_qpos = np.array([\n"
            names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3"]
            for i, val in enumerate(self.current_joints):
                new_qpos += f"            {val:7.4f},   # {names[i]}\n"
            new_qpos += "        ])"

            # Find and replace home_qpos section
            pattern = r'self\.home_qpos = np\.array\(\[.*?\]\)'

            if re.search(pattern, content, re.DOTALL):
                new_content = re.sub(pattern, new_qpos, content, flags=re.DOTALL)

                # Write back
                with open(env_file, 'w') as f:
                    f.write(new_content)

                print("\n" + "="*70)
                print("✅ SAVED TO ENVIRONMENT FILE!")
                print("="*70)
                print(f"File: {env_file}")
                print("\nNew home_qpos:")
                for i, name in enumerate(names):
                    deg = np.degrees(self.current_joints[i])
                    print(f"  {name:8s} (J{i+1}): {self.current_joints[i]:7.4f} rad ({deg:6.1f}°)")
                print("="*70)
            else:
                print("\n❌ Could not find home_qpos in environment file!")

        except Exception as e:
            print(f"\n❌ Error saving to file: {e}")
            print("Please copy manually using 'p' key")

    def can_do_action(self, action_name):
        """Check if enough time has passed since last action (debounce)."""
        current_time = time.time()
        last_time = self.last_action_time.get(action_name, 0)
        if current_time - last_time > self.action_cooldown:
            self.last_action_time[action_name] = current_time
            return True
        return False

    def on_press(self, key):
        """Callback for key press events."""
        with self.keys_lock:
            try:
                # Handle character keys
                if hasattr(key, 'char') and key.char:
                    self.keys_pressed.add(key.char)
            except AttributeError:
                # Handle special keys
                self.keys_pressed.add(key)

    def on_release(self, key):
        """Callback for key release events."""
        with self.keys_lock:
            try:
                # Handle character keys
                if hasattr(key, 'char') and key.char:
                    self.keys_pressed.discard(key.char)
            except AttributeError:
                # Handle special keys
                self.keys_pressed.discard(key)

    def is_key_pressed(self, key_name):
        """Check if a key is currently pressed."""
        with self.keys_lock:
            # Check for character keys
            if key_name in self.keys_pressed:
                return True
            # Check for special keys
            if key_name == 'space' and keyboard.Key.space in self.keys_pressed:
                return True
            if key_name == 'esc' and keyboard.Key.esc in self.keys_pressed:
                return True
            return False

    def process_keys(self):
        """Process currently pressed keys (called every frame)."""
        if not KEYBOARD_AVAILABLE:
            return

        # Continuous joint controls (hold key to keep moving)
        if self.is_key_pressed('1'):
            self.adjust_joint(0, -self.step_size)
        if self.is_key_pressed('q'):
            self.adjust_joint(0, self.step_size)

        if self.is_key_pressed('2'):
            self.adjust_joint(1, -self.step_size)
        if self.is_key_pressed('w'):
            self.adjust_joint(1, self.step_size)

        if self.is_key_pressed('3'):
            self.adjust_joint(2, -self.step_size)
        if self.is_key_pressed('e'):
            self.adjust_joint(2, self.step_size)

        if self.is_key_pressed('4'):
            self.adjust_joint(3, -self.step_size)
        if self.is_key_pressed('r'):
            self.adjust_joint(3, self.step_size)

        if self.is_key_pressed('5'):
            self.adjust_joint(4, -self.step_size)
        if self.is_key_pressed('t'):
            self.adjust_joint(4, self.step_size)

        if self.is_key_pressed('6'):
            self.adjust_joint(5, -self.step_size)
        if self.is_key_pressed('y'):
            self.adjust_joint(5, self.step_size)

        # Gripper
        if self.is_key_pressed('g'):
            self.adjust_gripper(self.gripper_step)
        if self.is_key_pressed('h'):
            self.adjust_gripper(-self.gripper_step)

        # Step size adjustment
        if self.is_key_pressed('['):
            if self.can_do_action('step_decrease'):
                self.step_size = max(0.005, self.step_size - 0.005)
                print(f"Step size: {self.step_size:.3f} rad")
        if self.is_key_pressed(']'):
            if self.can_do_action('step_increase'):
                self.step_size = min(0.1, self.step_size + 0.005)
                print(f"Step size: {self.step_size:.3f} rad")

        # Single-press actions (with debounce)
        if self.is_key_pressed('space'):
            if self.can_do_action('reset'):
                self.reset_to_home()

        if self.is_key_pressed('p'):
            if self.can_do_action('print'):
                self._print_code()

        if self.is_key_pressed('s'):
            if self.can_do_action('save'):
                self.save_home()

        if self.is_key_pressed('i'):
            if self.can_do_action('info'):
                self._print_info()

        if self.is_key_pressed('o'):
            if self.can_do_action('save_to_file'):
                self.save_to_env_file()

    def run(self):
        """Run teleoperation with MuJoCo viewer."""
        if not KEYBOARD_AVAILABLE:
            print("❌ 'pynput' library required for real-time control")
            print("Install with: pip install pynput")
            return

        print("\n🎮 실시간 제어 모드 시작!")
        print("   키를 누르고 있으면 로봇이 쭈욱 움직입니다 🚀\n")

        # Start keyboard listener in background thread
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

        try:
            with mujoco.viewer.launch_passive(
                self.env.model,
                self.env.data
            ) as viewer:
                # Set camera
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -20

                self._apply_state()

                while viewer.is_running():
                    # Check for ESC key to exit
                    if self.is_key_pressed('esc'):
                        print("\n👋 Exiting...")
                        break

                    # Process all pressed keys
                    self.process_keys()

                    # Update viewer
                    viewer.sync()

                    # Small delay to control update rate (~50 Hz)
                    time.sleep(0.02)
        finally:
            # Stop listener
            if self.listener:
                self.listener.stop()


def main():
    print("="*70)
    print("DSR H2017 실시간 Teleoperation")
    print("="*70)
    print("\n사용 방법:")
    print("  • 키를 누르고 있으면 로봇이 계속 움직입니다")
    print("  • Joint 제어: 1/q, 2/w, 3/e, 4/r, 5/t, 6/y")
    print("  • Gripper: g (close), h (open)")
    print("  • 정보: i (상태), p (코드 출력)")
    print("  • 저장: s (임시 저장), o (환경 파일에 자동 저장), SPACE (리셋)")
    print("  • 종료: ESC")
    print()

    controller = TeleopController()
    controller.run()
    controller.env.close()

    print("\n✅ Teleoperation 종료")


if __name__ == "__main__":
    main()
