from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse
import carb
import math
import numpy as np
import os
import queue
import threading
from pathlib import Path

import omni.appwindow

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.extensions import enable_extension
enable_extension('isaacsim.ros2.bridge')

from spot_policy import SpotFlatTerrainPolicy, SpotNavigationPolicy


BASE_DIR = Path(__file__).resolve().parent.parent


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _interpolate_yaw(start_yaw: float, end_yaw: float, alpha: float) -> float:
    delta = _wrap_to_pi(end_yaw - start_yaw)
    return _wrap_to_pi(start_yaw + alpha * delta)


def generate_waypoints(
    start: tuple, goal: tuple, max_step_xy: float
) -> list:
    """
    Linear XY interpolation with shortest-path yaw interpolation.
    start / goal: (x, y, yaw)
    Returns list of (x, y, yaw) waypoints including the final goal.
    """
    sx, sy, syaw = start
    gx, gy, gyaw = goal
    dist = math.hypot(gx - sx, gy - sy)
    if max_step_xy <= 0.0 or dist <= max_step_xy:
        return [goal]
    num_segments = math.ceil(dist / max_step_xy)
    waypoints = []
    for i in range(1, num_segments + 1):
        alpha = i / num_segments
        wx = sx + alpha * (gx - sx)
        wy = sy + alpha * (gy - sy)
        wyaw = _interpolate_yaw(syaw, gyaw, alpha)
        waypoints.append((wx, wy, wyaw))
    return waypoints


def input_worker(cmd_queue: queue.Queue, stop_event: threading.Event) -> None:
    """Background thread: reads goal / cancel / stop from stdin."""
    print("[Nav] Commands:")
    print("[Nav]   x y [yaw]  — set new goal  (e.g. '5.0 3.0 1.57', yaw defaults to 0)")
    print("[Nav]   c / cancel — cancel current goal and hold")
    print("[Nav]   stop       — stop motion and hold")
    print("[Nav]   quit       — exit simulation")
    while not stop_event.is_set():
        try:
            raw = input("goal> ").strip()
        except EOFError:
            break
        if not raw:
            continue
        token = raw.lower()
        if token in ("quit", "q", "exit"):
            cmd_queue.put(("quit", None))
            break
        if token in ("stop", "s"):
            cmd_queue.put(("stop", None))
            continue
        if token in ("cancel", "c"):
            cmd_queue.put(("cancel", None))
            continue
        parts = raw.split()
        if len(parts) in (2, 3):
            try:
                x, y = float(parts[0]), float(parts[1])
                yaw = float(parts[2]) if len(parts) == 3 else 0.0
                cmd_queue.put(("goal", (x, y, yaw)))
            except ValueError:
                print("[Nav] Invalid numbers. Example: 5.0 3.0 0.0")
        else:
            print("[Nav] Usage: x y [yaw]  |  c  |  stop  |  quit")


class SpotSensorsNavRunner:
    def __init__(self, physics_dt, render_dt, mode="teleop",
                 pos_thresh=0.3, yaw_thresh=0.3, waypoint_step=4.0):
        self._mode = mode
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        prim = define_prim("/World/Warehouse", "Xform")
        asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
        prim.GetReferences().AddReference(asset_path)

        policy_path = os.path.join(BASE_DIR, "policies/spot_flat/models", "policy.pt")
        params_path = os.path.join(BASE_DIR, "policies/spot_flat/params", "env.yaml")
        usd_path    = os.path.join(BASE_DIR, "assets", "spot_sensors.usd")

        self._spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=usd_path,
            policy_path=policy_path,
            policy_params_path=params_path,
            position=np.array([-8.0, 4.0, 0.8]),
            orientation=np.array([0.7071, 0.0, 0.0, 0.7071]),  # yaw=1.57 rad (w, x, y, z)
        )

        self._base_command = np.zeros(3)
        self.needs_reset = False
        self.first_step = True

        if mode == "teleop":
            self._speed      = 1.0
            self._speed_step = 0.5
            self._speed_min  = 0.5
            self._speed_max  = 3.0
            self._held_keys  = set()
            self._dir_mapping = {
                "NUMPAD_8": np.array([ 1.0,  0.0,  0.0]), "UP":    np.array([ 1.0,  0.0,  0.0]),
                "NUMPAD_2": np.array([-1.0,  0.0,  0.0]), "DOWN":  np.array([-1.0,  0.0,  0.0]),
                "NUMPAD_6": np.array([ 0.0, -1.0,  0.0]), "RIGHT": np.array([ 0.0, -1.0,  0.0]),
                "NUMPAD_4": np.array([ 0.0,  1.0,  0.0]), "LEFT":  np.array([ 0.0,  1.0,  0.0]),
                "NUMPAD_7": np.array([ 0.0,  0.0,  1.0]), "N":     np.array([ 0.0,  0.0,  1.0]),
                "NUMPAD_9": np.array([ 0.0,  0.0, -1.0]), "M":     np.array([ 0.0,  0.0, -1.0]),
            }

        elif mode == "nav":
            nav_policy_path = os.path.join(BASE_DIR, "policies/spot_nav/models", "policy.pt")
            self._nav_policy = SpotNavigationPolicy(nav_policy_path)
            print(f"[Nav] Navigation policy loaded. obs_dim={self._nav_policy.obs_dim}")

            # Waypoint state
            self._current_wp: tuple | None = None     # (x, y, yaw) currently tracking
            self._waypoint_queue: list     = []        # remaining waypoints
            self._stop_motion              = True
            self._goal_initialized         = False

            # Arrival thresholds
            self._pos_thresh    = pos_thresh
            self._yaw_thresh    = yaw_thresh
            self._waypoint_step = waypoint_step

            # Policy / print counters
            self._nav_policy_counter = 0
            self._nav_decimation     = 100  # 0.002s * 100 = 0.2s (5 Hz, matches training)
            self._print_counter      = 0

            # Inter-thread command queue
            self._cmd_queue  = queue.Queue()
            self._stop_event = threading.Event()
            self._running    = True

    # ------------------------------------------------------------------ #
    #  Setup
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input     = carb.input.acquire_input_interface()
        self._keyboard  = self._appwindow.get_keyboard()

        if self._mode == "teleop":
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(
                self._keyboard, self._sub_keyboard_event
            )
            self._world.add_physics_callback("spot_forward", callback_fn=self._on_physics_step_teleop)

        elif self._mode == "nav":
            self._world.add_physics_callback("spot_forward", callback_fn=self._on_physics_step_nav)
            t = threading.Thread(
                target=input_worker,
                args=(self._cmd_queue, self._stop_event),
                daemon=True,
            )
            t.start()

    # ------------------------------------------------------------------ #
    #  Teleop mode
    # ------------------------------------------------------------------ #

    def _recompute_command(self):
        cmd = np.zeros(3)
        for key in self._held_keys:
            cmd += self._dir_mapping[key] * self._speed
        cmd[0] = np.clip(cmd[0], -2.0, 3.0)
        cmd[1] = np.clip(cmd[1], -1.5, 1.5)
        cmd[2] = np.clip(cmd[2], -2.0, 2.0)
        self._base_command = cmd

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        key = event.input.name if hasattr(event.input, 'name') else event.input
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key in self._dir_mapping:
                self._held_keys.add(key)
                self._recompute_command()
                print(f"[CMD] vx={self._base_command[0]:+.1f}  vy={self._base_command[1]:+.1f}  wz={self._base_command[2]:+.1f}  speed={self._speed:.1f}")
            elif key in ("EQUAL", "NUMPAD_ADD"):
                self._speed = min(self._speed + self._speed_step, self._speed_max)
                self._recompute_command()
                print(f"[Speed] {self._speed:.1f} m/s")
            elif key in ("MINUS", "NUMPAD_SUBTRACT"):
                self._speed = max(self._speed - self._speed_step, self._speed_min)
                self._recompute_command()
                print(f"[Speed] {self._speed:.1f} m/s")
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key in self._dir_mapping:
                self._held_keys.discard(key)
                self._recompute_command()
        return True

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

    # ------------------------------------------------------------------ #
    #  Navigation mode — helpers
    # ------------------------------------------------------------------ #

    def _get_robot_pose(self) -> tuple:
        """Returns (x, y, yaw) in world frame."""
        pos_IB, q_IB = self._spot.robot.get_world_pose()
        w, x, y, z = q_IB[0], q_IB[1], q_IB[2], q_IB[3]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(pos_IB[0]), float(pos_IB[1]), yaw

    def _check_arrival(self) -> bool:
        if self._current_wp is None:
            return False
        rx, ry, ryaw = self._get_robot_pose()
        wx, wy, wyaw = self._current_wp
        dist    = math.hypot(rx - wx, ry - wy)
        yaw_err = abs(_wrap_to_pi(ryaw - wyaw))
        return dist < self._pos_thresh and yaw_err < self._yaw_thresh

    def _compute_pose_command(self, pos_IB, q_IB) -> np.ndarray:
        """Body-frame pose command for the navigation policy.

        Matches IsaacLab UniformPose2dCommand:
          pos_command_b  = R_yaw_inv @ (wp_pos_w - robot_pos_w)
          heading_command_b = wrap_to_pi(wp_yaw - robot_yaw)
        """
        wx, wy, wyaw = self._current_wp
        w, x, y, z = q_IB[0], q_IB[1], q_IB[2], q_IB[3]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        target_w   = np.array([wx, wy, pos_IB[2]])
        target_vec = target_w - pos_IB

        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
        R_yaw_inv = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ])
        pos_command_b = R_yaw_inv @ target_vec
        heading_error = _wrap_to_pi(wyaw - yaw)  # matches IsaacLab: heading_command_b = wrap_to_pi(wp_yaw - robot_yaw)

        pose_dim = self._nav_policy.obs_dim - 6
        if pose_dim == 4:
            return np.array([pos_command_b[0], pos_command_b[1], pos_command_b[2], heading_error])
        else:
            return np.array([pos_command_b[0], pos_command_b[1], heading_error])

    # ------------------------------------------------------------------ #
    #  Navigation mode — physics step
    # ------------------------------------------------------------------ #

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

        # --- Process incoming commands from stdin thread ---
        while True:
            try:
                cmd, payload = self._cmd_queue.get_nowait()
            except queue.Empty:
                break

            if cmd == "quit":
                print("\n[Nav] Quit received.")
                self._running = False
                return

            elif cmd == "stop":
                self._stop_motion = True
                self._waypoint_queue.clear()
                self._current_wp = None
                self._goal_initialized = False
                self._base_command = np.zeros(3)
                print("\n[Nav] Stopped. Holding current pose.")

            elif cmd == "cancel":
                self._stop_motion = True
                self._waypoint_queue.clear()
                self._current_wp = None
                self._goal_initialized = False
                self._base_command = np.zeros(3)
                print("\n[Nav] Goal cancelled.")

            elif cmd == "goal":
                current_pose = self._get_robot_pose()
                goal = payload  # (x, y, yaw)
                wps = generate_waypoints(current_pose, goal, self._waypoint_step)
                self._waypoint_queue = wps[1:]   # remaining after first
                self._current_wp = wps[0]
                self._stop_motion = False
                self._goal_initialized = True
                print(
                    f"\n[Nav] New goal ({goal[0]:.2f}, {goal[1]:.2f}, yaw={goal[2]:.2f}) | "
                    f"waypoints={len(wps)} first=({wps[0][0]:.2f}, {wps[0][1]:.2f}, {wps[0][2]:.2f})"
                )

        # --- Arrival check & waypoint advance ---
        if self._goal_initialized and not self._stop_motion:
            if self._check_arrival():
                if self._waypoint_queue:
                    self._current_wp = self._waypoint_queue.pop(0)
                    print(
                        f"\n[Nav] Waypoint reached → next: "
                        f"({self._current_wp[0]:.2f}, {self._current_wp[1]:.2f}, "
                        f"yaw={self._current_wp[2]:.2f})  remaining={len(self._waypoint_queue)}"
                    )
                else:
                    rx, ry, _ = self._get_robot_pose()
                    print(f"\n[Nav] Goal reached at ({rx:.2f}, {ry:.2f}). Holding.")
                    self._stop_motion = True
                    self._goal_initialized = False
                    self._base_command = np.zeros(3)

        # --- Hold if no goal or stopped ---
        if self._stop_motion or not self._goal_initialized:
            self._spot.forward(step_size, self._base_command)
            return

        # --- Run navigation policy ---
        if self._nav_policy_counter % self._nav_decimation == 0:
            lin_vel_I    = self._spot.robot.get_linear_velocity()
            pos_IB, q_IB = self._spot.robot.get_world_pose()
            R_BI         = quat_to_rot_matrix(q_IB).T

            lin_vel_b = np.matmul(R_BI, lin_vel_I)
            gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
            pose_cmd  = self._compute_pose_command(pos_IB, q_IB)

            nav_obs = np.concatenate([lin_vel_b, gravity_b, pose_cmd])
            raw_cmd = self._nav_policy.forward(nav_obs)

            self._base_command = np.array([
                np.clip(raw_cmd[0], -2.0, 3.0),
                np.clip(raw_cmd[1], -1.5, 1.5),
                np.clip(raw_cmd[2], -2.0, 2.0),
            ])

            self._print_counter += 1
            if self._print_counter % 2 == 0:
                wx, wy, _ = self._current_wp
                dist = math.hypot(pos_IB[0] - wx, pos_IB[1] - wy)
                print(
                    f"\r[Nav] pos=({pos_IB[0]:.2f},{pos_IB[1]:.2f}) "
                    f"wp=({wx:.2f},{wy:.2f}) dist={dist:.2f}m "
                    f"remaining_wp={len(self._waypoint_queue)} | "
                    f"cmd=[{self._base_command[0]:.2f},{self._base_command[1]:.2f},{self._base_command[2]:.2f}]",
                    end="", flush=True,
                )

        self._nav_policy_counter += 1
        self._spot.forward(step_size, self._base_command)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        while simulation_app.is_running() and self._running if self._mode == "nav" else simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True

    def shutdown(self):
        if self._mode == "nav":
            self._stop_event.set()
            self._running = False


def main():
    parser = argparse.ArgumentParser(description="Spot Sensors Navigation Demo")
    parser.add_argument(
        "--mode", choices=["teleop", "nav"], default="teleop",
        help="teleop: keyboard control | nav: waypoint-based autonomous navigation",
    )
    parser.add_argument("--pos-thresh",    type=float, default=0.3,
                        help="Arrival XY threshold in meters (default: 0.3)")
    parser.add_argument("--yaw-thresh",    type=float, default=0.3,
                        help="Arrival yaw threshold in radians (default: 0.3)")
    parser.add_argument("--waypoint-step", type=float, default=4.0,
                        help="Max XY distance per waypoint segment in meters (default: 4.0)")
    args, _ = parser.parse_known_args()

    physics_dt = 0.002  # matches training (dt=0.002, decimation=10 → LL at 50 Hz)
    render_dt  = 1 / 60.0

    print(f"[Spot Sensors Nav Demo] mode={args.mode}")
    runner = SpotSensorsNavRunner(
        physics_dt=physics_dt,
        render_dt=render_dt,
        mode=args.mode,
        pos_thresh=args.pos_thresh,
        yaw_thresh=args.yaw_thresh,
        waypoint_step=args.waypoint_step,
    )
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()

    if args.mode == "teleop":
        print("[Teleop] Controls: Arrow keys / Numpad = fwd/back/left/right | N/M = rotate CW/CCW")
        print("[Teleop] Speed:    = / NUMPAD+ = speed up  |  - / NUMPAD- = speed down  (0.5 ~ 3.0 m/s, default 1.0)")
        print("[Teleop] Click the Isaac Sim viewport first to give it keyboard focus.")

    runner.run()
    runner.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
