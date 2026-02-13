"""
Interactive Navigation with Go2.

Loads a trained locomotion policy and uses a high-level planner to navigate
to user-specified target positions. The robot walks to targets using the
trained velocity-tracking policy.

Usage:
    python navigate.py --model checkpoints/<run>/best_model.zip
    python navigate.py --model checkpoints/<run>/best_model.zip --terrain
"""

import os
import sys
import argparse
import time
import threading

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(__file__))
from envs.go2_env import Go2Env, NUM_JOINTS
from envs.go2_terrain_env import Go2TerrainEnv


class NavigationPlanner:
    """High-level planner: target position → velocity commands.

    Computes (vx, vy, yaw_rate) commands to drive the robot toward a target.
    Uses proportional control on distance and angular error.
    """

    def __init__(self, arrival_threshold=0.3, max_speed=0.8):
        self.target = None
        self.arrival_threshold = arrival_threshold
        self.max_speed = max_speed
        self.arrived = False
        self._lock = threading.Lock()

    def set_target(self, x, y):
        with self._lock:
            self.target = np.array([x, y])
            self.arrived = False

    def clear_target(self):
        with self._lock:
            self.target = None
            self.arrived = False

    def compute_command(self, robot_pos, robot_yaw):
        """Compute velocity command to reach target.

        Args:
            robot_pos: (3,) robot base position [x, y, z]
            robot_yaw: robot heading angle (rad)

        Returns:
            (3,) velocity command [vx, vy, yaw_rate]
        """
        with self._lock:
            if self.target is None:
                return np.zeros(3)

            diff = self.target - robot_pos[:2]
            distance = np.linalg.norm(diff)

            if distance < self.arrival_threshold:
                if not self.arrived:
                    self.arrived = True
                    print(f"  >> Arrived! (distance: {distance:.2f}m)")
                return np.zeros(3)

            self.arrived = False
            target_angle = np.arctan2(diff[1], diff[0])
            yaw_error = self._wrap_angle(target_angle - robot_yaw)

            # Forward speed: proportional to distance, reduced when turning
            cmd_vx = min(distance * 0.5, self.max_speed)
            cmd_vx *= max(0.0, np.cos(yaw_error))

            # Yaw rate: proportional to angular error
            cmd_yaw = np.clip(yaw_error * 2.0, -0.5, 0.5)

            return np.array([cmd_vx, 0.0, cmd_yaw])

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


def get_robot_yaw(quat):
    """Extract yaw angle from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_model_and_vecnorm(model_path):
    """Load PPO model and find VecNormalize stats."""
    model = PPO.load(model_path, device="cpu")

    vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if not os.path.exists(vecnorm_path):
        model_dir = os.path.dirname(model_path)
        for f in os.listdir(model_dir):
            if f.endswith("_vecnormalize.pkl"):
                vecnorm_path = os.path.join(model_dir, f)
                break

    return model, vecnorm_path


def normalize_obs(obs, vec_env):
    """Normalize observation using VecNormalize statistics."""
    if isinstance(vec_env, VecNormalize):
        return vec_env.normalize_obs(obs.reshape(1, -1)).flatten()
    return obs


def add_markers(viewer, target_pos, robot_pos, terrain_z=0.0):
    """Draw target marker and path line in the viewer."""
    if target_pos is None:
        viewer.user_scn.ngeom = 0
        return

    # Target sphere (red)
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [0.1, 0, 0],
        [target_pos[0], target_pos[1], terrain_z + 0.15],
        np.eye(3).flatten(),
        [1.0, 0.2, 0.2, 0.8],
    )

    # Path line from robot to target (green capsule)
    mujoco.mjv_makeConnector(
        viewer.user_scn.geoms[1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        0.005,
        robot_pos[0], robot_pos[1], 0.05,
        target_pos[0], target_pos[1], terrain_z + 0.05,
    )
    viewer.user_scn.geoms[1].rgba[:] = [0.2, 1.0, 0.2, 0.5]

    viewer.user_scn.ngeom = 2


def input_thread_fn(planner, stop_event):
    """Thread for reading target position input from terminal."""
    print()
    print("=" * 45)
    print("  Go2 Navigation Controller")
    print("=" * 45)
    print("  Commands:")
    print("    x y       - Set target (e.g., '3 2')")
    print("    stop      - Clear target (stand still)")
    print("    quit      - Exit")
    print("=" * 45)
    print()

    while not stop_event.is_set():
        try:
            line = input("Target> ").strip()
            if not line:
                continue
            if line.lower() == "quit":
                stop_event.set()
                break
            elif line.lower() == "stop":
                planner.clear_target()
                print("  Target cleared.")
            else:
                parts = line.split()
                x, y = float(parts[0]), float(parts[1])
                planner.set_target(x, y)
                print(f"  -> Target set: ({x:.1f}, {y:.1f})")
        except (ValueError, IndexError):
            print("  Invalid. Use: x y (e.g., '3 2')")
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(description="Go2 Interactive Navigation")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.zip)"
    )
    parser.add_argument(
        "--terrain", action="store_true", help="Use terrain environment"
    )
    parser.add_argument(
        "--difficulty", type=float, default=0.3, help="Terrain difficulty (0-1)"
    )
    args = parser.parse_args()

    # Load trained policy
    model_ppo, vecnorm_path = load_model_and_vecnorm(args.model)
    print(f"Loaded policy from: {args.model}")

    # Create environment for normalization
    if args.terrain:
        make_env = lambda: Go2TerrainEnv(difficulty=args.difficulty)
    else:
        make_env = lambda: Go2Env()

    vec_env = DummyVecEnv([make_env])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize from: {vecnorm_path}")

    # Create raw environment for simulation + viewer
    if args.terrain:
        env = Go2TerrainEnv(difficulty=args.difficulty)
    else:
        env = Go2Env()

    obs, info = env.reset()

    # Extend episode length for navigation (no truncation during active navigation)
    env.max_episode_steps = 50000  # ~1000 seconds

    # Navigation planner
    planner = NavigationPlanner()
    stop_event = threading.Event()

    # Start input thread
    input_t = threading.Thread(
        target=input_thread_fn, args=(planner, stop_event), daemon=True
    )
    input_t.start()

    total_reward = 0.0
    step_count = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and not stop_event.is_set():
            step_start = time.time()

            # Get robot state
            robot_pos = env._get_base_pos()
            robot_yaw = get_robot_yaw(env._get_base_quat())

            # Navigation planner → velocity command
            cmd = planner.compute_command(robot_pos, robot_yaw)
            env._command = cmd

            # Get terrain height at target for marker placement
            target_terrain_z = 0.0
            with planner._lock:
                target = planner.target.copy() if planner.target is not None else None
                arrived = planner.arrived
            if target is not None and args.terrain:
                target_terrain_z = env._get_terrain_height_at(target[0], target[1])

            # Draw markers
            add_markers(viewer, target, robot_pos, target_terrain_z)

            # When arrived or no target: bypass policy, hold default stance
            if arrived or target is None:
                action = np.zeros(NUM_JOINTS, dtype=np.float32)
            else:
                # Policy inference
                obs_norm = normalize_obs(obs, vec_env)
                action, _ = model_ppo.predict(
                    obs_norm.reshape(1, -1), deterministic=True
                )
                action = action.flatten()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Print status periodically
            if step_count % 100 == 0 and target is not None:
                dist = np.linalg.norm(target - robot_pos[:2])
                print(
                    f"  [step {step_count}] pos=({robot_pos[0]:.1f}, {robot_pos[1]:.1f}) "
                    f"dist={dist:.2f}m cmd_vx={cmd[0]:.2f}"
                )

            # Reset if needed
            if terminated or truncated:
                print(
                    f"  Episode reset: steps={step_count}, reward={total_reward:.1f}"
                )
                obs, info = env.reset()
                total_reward = 0.0
                step_count = 0

            # Sync viewer
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            sleep_time = env.control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    stop_event.set()
    env.close()
    vec_env.close()
    print("\nNavigation session ended.")


if __name__ == "__main__":
    main()
