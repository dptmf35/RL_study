from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse
import carb
import numpy as np
import os
import threading
from pathlib import Path

import omni.appwindow

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.storage.native import get_assets_root_path
from spot_policy import SpotFlatTerrainPolicy, SpotNavigationPolicy


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class SpotNavRunner:
    def __init__(self, physics_dt, render_dt, mode="teleop"):
        self._mode = mode
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # Spawn warehouse
        prim = define_prim("/World/Warehouse", "Xform")
        asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        prim.GetReferences().AddReference(asset_path)

        BASE_DIR = Path(__file__).resolve().parent.parent
        loco_policy_path = os.path.join(BASE_DIR, "policies/spot_flat/models", "policy.pt")
        loco_params_path = os.path.join(BASE_DIR, "policies/spot_flat/params", "env.yaml")
        usd_path = os.path.join(BASE_DIR, "assets", "spot.usd")

        self._spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=usd_path,
            policy_path=loco_policy_path,
            policy_params_path=loco_params_path,
            position=np.array([1, 0, 0.8]),
        )

        self._base_command = np.zeros(3)
        self.needs_reset = False
        self.first_step = True

        if mode == "teleop":
            self._input_keyboard_mapping = {
                "NUMPAD_8": [1.0, 0.0, 0.0], "UP":    [1.0, 0.0, 0.0],
                "NUMPAD_2": [-1.0, 0.0, 0.0], "DOWN": [-1.0, 0.0, 0.0],
                "NUMPAD_6": [0.0, -1.0, 0.0], "RIGHT": [0.0, -1.0, 0.0],
                "NUMPAD_4": [0.0, 1.0, 0.0],  "LEFT":  [0.0, 1.0, 0.0],
                "NUMPAD_7": [0.0, 0.0, 1.0],  "N": [0.0, 0.0, 1.0],
                "NUMPAD_9": [0.0, 0.0, -1.0], "M": [0.0, 0.0, -1.0],
            }
        elif mode == "nav":
            nav_policy_path = os.path.join(BASE_DIR, "policies/spot_nav/models", "policy.pt")
            self._nav_policy = SpotNavigationPolicy(nav_policy_path)
            print(f"[Nav] Navigation policy loaded. obs_dim={self._nav_policy.obs_dim}")
            self._nav_target = None
            self._nav_policy_counter = 0
            self._nav_decimation = 100
            self._running = True
            self._print_counter = 0

    def setup(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        if self._mode == "teleop":
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(
                self._keyboard, self._sub_keyboard_event
            )
            self._world.add_physics_callback("spot_forward", callback_fn=self._on_physics_step_teleop)
        elif self._mode == "nav":
            self._world.add_physics_callback("spot_forward", callback_fn=self._on_physics_step_nav)
            t = threading.Thread(target=self._target_input_thread, daemon=True)
            t.start()
            print("[Nav] Waiting for target. Type 'x y' and press Enter.")

    # ------------------------------------------------------------------ #
    #  Teleop mode
    # ------------------------------------------------------------------ #

    def _on_physics_step_teleop(self, step_size) -> None:
        if self.first_step:
            self._spot.initialize()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            self._spot.forward(step_size, self._base_command)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    # ------------------------------------------------------------------ #
    #  Navigation mode
    # ------------------------------------------------------------------ #

    def _target_input_thread(self):
        """Background thread: reads target (x, y) from stdin."""
        while self._running:
            try:
                line = input("Enter target x y: ")
                parts = line.strip().split()
                if len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    self._nav_target = np.array([x, y])
                    print(f"[Nav] New target set: ({x:.2f}, {y:.2f})")
                else:
                    print("[Nav] Usage: enter two numbers, e.g.  5.0 3.0")
            except (ValueError, EOFError):
                pass

    def _compute_pose_command(self, pos_IB, q_IB) -> np.ndarray:
        """
        Compute pose_command in body frame matching IsaacLab's UniformPose2dCommand.
        Uses yaw-only rotation (same as training: quat_apply_inverse(yaw_quat(...))).

        Returns np.ndarray of shape (3,) or (4,) matching self._nav_policy.obs_dim - 6.
        """
        # Extract yaw from quaternion (w, x, y, z convention in Isaac Sim)
        w, x, y, z = q_IB[0], q_IB[1], q_IB[2], q_IB[3]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Target vector in world frame
        target_w = np.array([self._nav_target[0], self._nav_target[1], pos_IB[2]])
        target_vec = target_w - pos_IB

        # Rotate to body frame using inverse yaw rotation
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        R_yaw_inv = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ])
        pos_command_b = R_yaw_inv @ target_vec

        # Heading error: point toward target
        target_heading = np.arctan2(target_vec[1], target_vec[0])
        heading_error = _wrap_to_pi(target_heading - yaw)

        pose_dim = self._nav_policy.obs_dim - 6  # 6 = lin_vel(3) + gravity(3)
        if pose_dim == 4:
            return np.array([pos_command_b[0], pos_command_b[1], pos_command_b[2], heading_error])
        else:  # 3
            return np.array([pos_command_b[0], pos_command_b[1], heading_error])

    def _on_physics_step_nav(self, step_size) -> None:
        if self.first_step:
            self._spot.initialize()
            self.first_step = False
            return
        if self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
            return

        # No target yet: do nothing (robot holds default pose)
        if self._nav_target is None:
            return

        # Run navigation policy at its decimation rate
        if self._nav_policy_counter % self._nav_decimation == 0:
            lin_vel_I = self._spot.robot.get_linear_velocity()
            pos_IB, q_IB = self._spot.robot.get_world_pose()
            R_IB = quat_to_rot_matrix(q_IB)
            R_BI = R_IB.transpose()

            lin_vel_b = np.matmul(R_BI, lin_vel_I)
            gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
            pose_cmd = self._compute_pose_command(pos_IB, q_IB)

            nav_obs = np.concatenate([lin_vel_b, gravity_b, pose_cmd])
            raw_cmd = self._nav_policy.forward(nav_obs)

            # Clamp to locomotion policy training range
            self._base_command = np.array([
                np.clip(raw_cmd[0], -2.0, 3.0),   # v_x
                np.clip(raw_cmd[1], -1.5, 1.5),   # v_y
                np.clip(raw_cmd[2], -2.0, 2.0),   # w_z
            ])

            # Print distance every ~1s (nav policy at 2 Hz -> every 2 calls)
            self._print_counter += 1
            if self._print_counter % 2 == 0:
                dist = np.linalg.norm(self._nav_target - pos_IB[:2])
                print(
                    f"\r[Nav] pos=({pos_IB[0]:.2f},{pos_IB[1]:.2f}) "
                    f"target=({self._nav_target[0]:.2f},{self._nav_target[1]:.2f}) "
                    f"dist={dist:.2f}m | "
                    f"raw=[{raw_cmd[0]:.2f},{raw_cmd[1]:.2f},{raw_cmd[2]:.2f}] "
                    f"cmd=[{self._base_command[0]:.2f},{self._base_command[1]:.2f},{self._base_command[2]:.2f}]",
                    end="", flush=True,
                )

        self._nav_policy_counter += 1

        # Locomotion policy handles its own decimation internally
        self._spot.forward(step_size, self._base_command)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True

    def shutdown(self):
        if self._mode == "nav":
            self._running = False


def main():
    parser = argparse.ArgumentParser(description="Spot Navigation Demo")
    parser.add_argument(
        "--mode",
        choices=["teleop", "nav"],
        default="teleop",
        help="teleop: keyboard control | nav: autonomous navigation to typed coordinates",
    )
    # argparse must be parsed before SimulationApp consumes sys.argv
    args, _ = parser.parse_known_args()

    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    print(f"[Spot Nav Demo] mode={args.mode}")
    runner = SpotNavRunner(physics_dt=physics_dt, render_dt=render_dt, mode=args.mode)
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()

    if args.mode == "teleop":
        print("[Teleop] Controls: Arrow keys / Numpad = fwd/back/left/right | N/M = rotate CW/CCW")

    runner.run()
    runner.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
