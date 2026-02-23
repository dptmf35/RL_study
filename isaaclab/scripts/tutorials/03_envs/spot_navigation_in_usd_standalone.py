# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone Isaac Sim inference for Spot navigation in warehouse USD.

This script does not use rsl_rl/play.py. It launches Isaac Sim, builds a ManagerBasedRLEnv directly,
loads an exported JIT policy, then runs goal-conditioned navigation.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import io
import os
import queue
import sys
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Standalone Spot navigation inference in warehouse USD.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to high-level navigation JIT policy.pt (exported).",
)
parser.add_argument(
    "--low-level-policy",
    type=str,
    default=None,
    help="Path to Spot low-level JIT policy.pt. Overrides ISAACLAB_SPOT_LOW_LEVEL_POLICY_PATH if set.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--pos-thresh", type=float, default=0.25, help="Arrival threshold for XY position error (meters).")
parser.add_argument("--yaw-thresh", type=float, default=0.25, help="Arrival threshold for yaw error (radians).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.navigation.config.spot.navigation_env_cfg import NavigationSpotWarehouseEnvCfg_PLAY


def input_worker(cmd_queue: queue.Queue, stop_event: threading.Event) -> None:
    """Receive runtime commands from terminal without blocking the simulation loop."""
    if not sys.stdin.isatty():
        print("[WARN] Non-interactive stdin detected. Runtime goal updates are disabled.")
        return

    print("[INPUT] Runtime commands:")
    print("[INPUT]   - First goal required: x y yaw   (example: 1.0 -2.5 0.0)")
    print("[INPUT]   - Stop now: stop")
    print("[INPUT]   - Quit app: quit")
    while not stop_event.is_set():
        try:
            raw = input("goal> ").strip()
        except EOFError:
            break
        if raw == "":
            continue
        token = raw.lower()
        if token in ("quit", "q", "exit"):
            cmd_queue.put(("quit", None))
            break
        if token in ("stop", "s"):
            cmd_queue.put(("stop", None))
            continue
        parts = raw.split()
        if len(parts) != 3:
            print("[WARN] Use `x y yaw`, `stop`, or `quit`.")
            continue
        try:
            goal = (float(parts[0]), float(parts[1]), float(parts[2]))
            cmd_queue.put(("goal", goal))
        except ValueError:
            print("[WARN] Invalid numbers. Example: 1.0 -2.5 0.0")


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def set_fixed_goal(env: ManagerBasedRLEnv, goal_x: float, goal_y: float, goal_yaw: float) -> None:
    """Overwrite `pose_command` term in world frame and block resampling."""
    term = env.command_manager.get_term("pose_command")
    term.pos_command_w[:, 0] = goal_x
    term.pos_command_w[:, 1] = goal_y
    term.pos_command_w[:, 2] = term.robot.data.default_root_state[:, 2]
    term.heading_command_w[:] = goal_yaw
    term.time_left[:] = 1.0e9


def compute_goal_error(env: ManagerBasedRLEnv, goal_x: float, goal_y: float, goal_yaw: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env XY/yaw errors in world frame."""
    term = env.command_manager.get_term("pose_command")
    root_xy = term.robot.data.root_pos_w[:, :2]
    heading = term.robot.data.heading_w

    goal_xy = torch.tensor([goal_x, goal_y], device=env.device, dtype=root_xy.dtype).unsqueeze(0)
    pos_error = torch.linalg.norm(root_xy - goal_xy, dim=1)
    goal_yaw_t = torch.full_like(heading, fill_value=goal_yaw)
    yaw_error = torch.abs(wrap_to_pi(heading - goal_yaw_t))
    return pos_error, yaw_error


def main():
    # Load high-level JIT policy.
    policy_path = os.path.abspath(args_cli.checkpoint)
    try:
        file_content = omni.client.read_file(policy_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        policy = torch.jit.load(file, map_location=args_cli.device)
        policy.eval()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load high-level JIT policy. "
            "Use an exported TorchScript file such as "
            "'logs/rsl_rl/spot_navigation/<run>/exported/policy.pt', "
            "not raw training checkpoints like 'model_XXXX.pt'."
        ) from exc

    env_cfg = NavigationSpotWarehouseEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # Force deterministic spawn: robot starts at (0, 0, yaw=0), zero velocity.
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    if args_cli.low_level_policy is not None:
        env_cfg.actions.pre_trained_policy_action.policy_path = os.path.abspath(args_cli.low_level_policy)

    print("[INFO] Creating standalone Isaac Sim environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs, _ = env.reset()
    goal_x, goal_y, goal_yaw = 0.0, 0.0, 0.0
    set_fixed_goal(env, goal_x, goal_y, goal_yaw)
    print("[INFO] Robot is initialized at origin and will hold position until first goal is entered.")
    print(f"[INFO] Arrival thresholds: pos_th={args_cli.pos_thresh:.3f}, yaw_th={args_cli.yaw_thresh:.3f}")

    step_count = 0
    arrived_once = False
    stop_motion = True
    goal_initialized = False
    cmd_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(target=input_worker, args=(cmd_queue, stop_event), daemon=True)
    thread.start()

    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                while True:
                    try:
                        cmd, payload = cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    if cmd == "quit":
                        print("[INFO] Quit command received. Closing simulation.")
                        return
                    if cmd == "stop":
                        stop_motion = True
                        arrived_once = False
                        print("[INFO] Stop command received. Holding zero action.")
                        continue
                    if cmd == "goal":
                        goal_x, goal_y, goal_yaw = payload
                        goal_initialized = True
                        stop_motion = False
                        arrived_once = False
                        print(
                            "[INFO] New goal applied: "
                            f"x={goal_x:.3f}, y={goal_y:.3f}, yaw={goal_yaw:.3f} rad"
                        )

                set_fixed_goal(env, goal_x, goal_y, goal_yaw)
                pos_error, yaw_error = compute_goal_error(env, goal_x, goal_y, goal_yaw)
                arrived = (pos_error < args_cli.pos_thresh) & (yaw_error < args_cli.yaw_thresh)

                if (not goal_initialized) or stop_motion or torch.any(arrived):
                    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                    if (not stop_motion) and (not arrived_once):
                        print(
                            f"[INFO] Arrival detected at step={step_count}: "
                            f"pos_err={pos_error[0].item():.3f} m, yaw_err={yaw_error[0].item():.3f} rad"
                        )
                        arrived_once = True
                else:
                    actions = policy(obs["policy"])

                obs, _, _, _, _ = env.step(actions)

                if step_count % 100 == 0:
                    print(
                        f"[INFO] step={step_count:05d} pos_err={pos_error[0].item():.3f} m "
                        f"yaw_err={yaw_error[0].item():.3f} rad arrived={bool(arrived[0].item())} "
                        f"stop={stop_motion} waiting_goal={not goal_initialized}"
                    )
                step_count += 1
    finally:
        stop_event.set()
        thread.join(timeout=0.5)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
