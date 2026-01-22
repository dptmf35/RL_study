"""
UR5e + Robotiq 2F85 Pick and Place Environment for Reinforcement Learning

This environment simulates a UR5e robot arm with Robotiq 2F85 gripper performing
pick and place tasks with a cube object.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class UR5eRobotiqPickPlaceEnv(gym.Env):
    """
    UR5e + Robotiq 2F85 Pick and Place Environment.

    Observation Space:
        - Joint positions (6)
        - Joint velocities (6)
        - Gripper position (1) - driver joint position
        - Gripper velocity (1)
        - End effector position (3)
        - Cube position (3)
        - Cube velocity (3)
        - Target position (3)
        - Relative cube to gripper (3)
        - Relative cube to target (3)
        Total: 32 dimensions

    Action Space:
        - Delta joint positions (6) - continuous [-1, 1]
        - Gripper command (1) - continuous [-1, 1] (mapped to 0-255)
        Total: 7 dimensions

    Reward:
        - Distance-based reward for reaching the cube
        - Grasping reward when cube is lifted
        - Placement reward when cube reaches target
        - Task completion bonus
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        reward_type: str = "dense",
        randomize_cube: bool = True,
        randomize_target: bool = False,
        task_mode: str = "pick_place",  # "reach", "pick", "pick_place"
        easy_mode: bool = False,
    ):
        """
        Initialize the UR5e + Robotiq 2F85 Pick and Place environment.

        Args:
            render_mode: "human" for live rendering, "rgb_array" for image output
            max_episode_steps: Maximum steps per episode
            reward_type: "dense" or "sparse" reward
            randomize_cube: Whether to randomize cube initial position
            randomize_target: Whether to randomize target position
            task_mode: "reach", "pick", "pick_place"
            easy_mode: If True, use smaller randomization range
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.randomize_cube = randomize_cube
        self.randomize_target = randomize_target
        self.task_mode = task_mode
        self.easy_mode = easy_mode

        # Load MuJoCo model
        model_path = Path(__file__).parent.parent / "assets" / "ur5e_robotiq_pick_place.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.frame_skip = 10

        # Robot configuration
        self.n_joints = 6
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # Get joint indices
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                         for name in self.joint_names]
        
        # Robotiq gripper - uses driver joint position
        self.gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")

        # Get actuator indices
        self.actuator_ids = list(range(6))  # First 6 actuators are for arm
        self.gripper_actuator_id = 6  # Robotiq gripper actuator

        # Get site/body indices
        self.pinch_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32
        )

        # Action scaling - larger scale for faster movement
        self.joint_action_scale = 0.15
        self.gripper_action_threshold = 0.0  # Threshold for gripper open/close

        # Home position - arm extended towards table, gripper pointing down
        # [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
        self.home_qpos = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

        # Workspace limits for cube spawning (relative to table at 0.5, 0, 0.4)
        # Cube is now 2.5cm, so z = 0.42 (table) + 0.025 (half cube) = 0.445
        if self.easy_mode:
            # VERY easy: cube spawns right in front of robot, minimal randomization
            self.cube_spawn_range = {
                "x": (0.30, 0.35),  # Close to robot
                "y": (-0.03, 0.03),  # Minimal Y variation
                "z": 0.445  # On table surface
            }
        else:
            self.cube_spawn_range = {
                "x": (0.3, 0.45),
                "y": (-0.1, 0.1),
                "z": 0.445
            }
        
        self.target_position = np.array([0.35, 0.15, 0.42])  # Closer target

        # Task thresholds
        self.grasp_height_threshold = 0.45  # Height to consider cube grasped
        self.place_threshold = 0.05
        self.reach_threshold = 0.10  # 10cm - reasonable distance for gripper approach

        # Episode tracking
        self.current_step = 0
        self.task_success = False

        # Rendering
        self.viewer = None
        self._render_context = None

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Joint positions and velocities
        joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                             for jid in self.joint_ids])
        joint_vel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]]
                             for jid in self.joint_ids])

        # Gripper state (driver joint position and velocity)
        gripper_pos = self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]
        gripper_vel = self.data.qvel[self.model.jnt_dofadr[self.gripper_joint_id]]

        # End effector position (pinch site)
        ee_pos = self.data.site_xpos[self.pinch_site_id].copy()

        # Cube position and velocity
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        cube_vel = self._get_cube_velocity()

        # Target position
        target_pos = self.data.site_xpos[self.target_site_id].copy()

        # Relative positions
        cube_to_gripper = cube_pos - ee_pos
        cube_to_target = cube_pos - target_pos

        obs = np.concatenate([
            joint_pos,        # 6
            joint_vel,        # 6
            [gripper_pos],    # 1
            [gripper_vel],    # 1
            ee_pos,           # 3
            cube_pos,         # 3
            cube_vel,         # 3
            target_pos,       # 3
            cube_to_gripper,  # 3
            cube_to_target,   # 3
        ]).astype(np.float32)

        return obs

    def _get_cube_velocity(self) -> np.ndarray:
        """Get cube linear velocity."""
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        vel_addr = self.model.jnt_dofadr[cube_joint_id]
        return self.data.qvel[vel_addr:vel_addr+3].copy()

    def _check_grasp(self) -> bool:
        """
        Check if cube is grasped using physics-based detection.
        
        Returns True if:
        1. Gripper is closed (gripper_pos > 0.4)
        2. Cube is close to gripper (< 6cm)
        3. Cube is lifted above table (> 2cm)
        
        This is stricter than just checking distance.
        """
        ee_pos = self.data.site_xpos[self.pinch_site_id]
        cube_pos = self.data.xpos[self.cube_body_id]
        gripper_pos = self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]
        
        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        gripper_is_closed = gripper_pos > 0.4
        cube_height = cube_pos[2]
        table_height = 0.44
        cube_is_lifted = cube_height > (table_height + 0.02)
        
        # All three conditions must be met
        return gripper_is_closed and dist_to_cube < 0.06 and cube_is_lifted

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        ee_pos = self.data.site_xpos[self.pinch_site_id]
        cube_pos = self.data.xpos[self.cube_body_id]
        target_pos = self.data.site_xpos[self.target_site_id]

        # Check if cube is grasped using physics
        is_grasped = self._check_grasp()

        # Check if cube is at target
        dist_to_target = np.linalg.norm(cube_pos - target_pos)
        is_obj_placed = dist_to_target < self.place_threshold

        # Check if robot is static
        joint_vel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]]
                             for jid in self.joint_ids])
        is_robot_static = np.linalg.norm(joint_vel) < 0.2

        return {
            "distance_to_cube": np.linalg.norm(ee_pos - cube_pos),
            "distance_to_target": dist_to_target,
            "cube_height": cube_pos[2],
            "is_grasped": is_grasped,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "task_success": self.task_success,
            "current_step": self.current_step,
        }

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward using PickCube-style Staged/Gated Reward pattern.
        
        Key principles from ManiSkill PickCube:
        1. Use tanh(5 * distance) for smooth distance rewards
        2. Use is_grasped as a KEY/MASK for place reward
        3. Use is_obj_placed as a MASK for static reward
        4. Give MAXIMUM reward for success to dominate all other rewards
        """
        ee_pos = self.data.site_xpos[self.pinch_site_id].copy()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        target_pos = self.data.site_xpos[self.target_site_id].copy()

        # Get info for gating
        info = self._get_info()
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        is_robot_static = info["is_robot_static"]

        # Distances
        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_to_target = np.linalg.norm(cube_pos - target_pos)

        reward_components = {
            "reaching": 0.0,
            "grasped": 0.0,
            "placing": 0.0,
            "static": 0.0,
            "success": 0.0,
        }

        # === REACH TASK ===
        if self.task_mode == "reach":
            if self.reward_type == "sparse":
                # Fetch-style sparse reward: -1 if not reached, 0 if reached
                if dist_to_cube < self.reach_threshold:
                    reward_components["success"] = 0.0
                    self.task_success = True
                else:
                    reward_components["reaching"] = -1.0
            else:
                # Dense reward: negative distance (Fetch-style dense)
                reward_components["reaching"] = -dist_to_cube

                # Success bonus
                if dist_to_cube < self.reach_threshold:
                    reward_components["success"] = 1.0
                    self.task_success = True

            total_reward = sum(reward_components.values())
            return total_reward, reward_components

        # === PICK TASK ===
        elif self.task_mode == "pick":
            # Get gripper state
            gripper_pos = self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]
            gripper_is_open = gripper_pos < 0.2  # Open when joint < 0.2
            gripper_is_closed = gripper_pos > 0.5  # Closed when joint > 0.5

            # Stage 1: Reaching reward (always active)
            reaching_reward = 1 - np.tanh(5 * dist_to_cube)
            reward_components["reaching"] = reaching_reward

            # Stage 2: Gripper preparation - CRITICAL for learning grasp sequence
            # When close to cube (< 10cm), reward for having gripper OPEN (ready to grasp)
            if dist_to_cube < 0.10 and not is_grasped:
                if gripper_is_open:
                    reward_components["gripper_prep"] = 0.5  # Reward for open gripper near cube
                else:
                    reward_components["gripper_prep"] = -0.2  # Penalize closed gripper when approaching

            # Stage 3: Grasped bonus (binary) - much bigger reward
            if is_grasped:
                reward_components["grasped"] = 2.0  # Strong reward for successful grasp

            # Stage 4: Lifting reward (only when grasped)
            cube_height = cube_pos[2]
            table_height = 0.44
            if is_grasped:
                lift_height = max(0, cube_height - table_height)
                reward_components["lifting"] = min(lift_height * 10.0, 1.0)  # Up to 1.0 for 10cm lift

            # Success: cube grasped and lifted
            if is_grasped and cube_height > (table_height + 0.08):
                reward_components["success"] = 5.0  # Max reward
                self.task_success = True

            total_reward = sum(reward_components.values())
            return total_reward, reward_components

        # === PICK AND PLACE TASK (PickCube style) ===
        else:  # pick_place
            # Stage 1: Reaching reward (always active)
            # 1 - tanh(5 * dist): dist=0 → 1.0, dist=0.2 → 0.37, dist=∞ → 0
            reaching_reward = 1 - np.tanh(5 * dist_to_cube)
            reward_components["reaching"] = reaching_reward

            # Stage 2: Grasped bonus (binary)
            # CRITICAL: This is just a bonus, NOT a gate yet
            if is_grasped:
                reward_components["grasped"] = 1.0

            # Stage 3: Placing reward (GATED by is_grasped)
            # This is the KEY pattern from PickCube!
            # If not grasped, place_reward = 0 (masked out)
            # If grasped, place_reward encourages moving to target
            place_reward = 1 - np.tanh(5 * dist_to_target)
            reward_components["placing"] = place_reward * float(is_grasped)  # GATING!

            # Stage 4: Static reward (GATED by is_obj_placed)
            # Only reward being static when cube is at target
            # This prevents "throwing" the cube and getting instant success
            joint_vel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]]
                                 for jid in self.joint_ids])
            qvel_norm = np.linalg.norm(joint_vel)
            static_reward = 1 - np.tanh(5 * qvel_norm)
            reward_components["static"] = static_reward * float(is_obj_placed)  # GATING!

            # Success: cube at target AND robot is static
            # Max reward to dominate all staged rewards
            if is_obj_placed and is_robot_static:
                reward_components["success"] = 5.0
                self.task_success = True

            total_reward = sum(reward_components.values())
            return total_reward, reward_components

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set robot to home position
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = self.home_qpos[i]

        # Open gripper (set driver joints to initial position)
        gripper_qpos_addr = self.model.jnt_qposadr[self.gripper_joint_id]
        self.data.qpos[gripper_qpos_addr] = 0.0  # Open position

        # Randomize cube position
        if self.randomize_cube:
            cube_x = self.np_random.uniform(*self.cube_spawn_range["x"])
            cube_y = self.np_random.uniform(*self.cube_spawn_range["y"])
            cube_z = self.cube_spawn_range["z"]
        else:
            cube_x, cube_y, cube_z = 0.32, 0.0, 0.445  # Fixed position close to robot

        # Set cube position
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1, 0, 0, 0]

        # Randomize target if needed
        if self.randomize_target:
            target_x = self.np_random.uniform(0.4, 0.6)
            target_y = self.np_random.uniform(-0.1, 0.1)
            self.target_position = np.array([target_x, target_y, 0.42])

        # Reset episode tracking
        self.current_step = 0
        self.task_success = False

        # Forward simulation
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.current_step += 1

        # Parse action
        joint_action = action[:6] * self.joint_action_scale
        gripper_action = action[6]

        # Apply joint actions (delta control)
        current_joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                                     for jid in self.joint_ids])
        target_joint_pos = current_joint_pos + joint_action

        # Clip to joint limits
        for i, jid in enumerate(self.joint_ids):
            jnt_range = self.model.jnt_range[jid]
            target_joint_pos[i] = np.clip(target_joint_pos[i], jnt_range[0], jnt_range[1])

        # Set control targets
        self.data.ctrl[:6] = target_joint_pos

        # Gripper control: map [-1, 1] to [0, 0.8]
        # -1 = open (0), 1 = close (0.8)
        gripper_ctrl = (gripper_action + 1.0) * 0.4  # Map to 0-0.8
        gripper_ctrl = np.clip(gripper_ctrl, 0, 0.8)
        self.data.ctrl[6] = gripper_ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Get observation and reward
        obs = self._get_obs()
        reward, reward_info = self._compute_reward()
        info = self._get_info()
        info["reward_components"] = reward_info

        # Check termination
        terminated = self.task_success
        truncated = self.current_step >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None

        elif self.render_mode == "rgb_array":
            if self._render_context is None:
                self._render_context = mujoco.Renderer(self.model, height=480, width=640)
            self._render_context.update_scene(self.data)
            return self._render_context.render()

        return None

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._render_context is not None:
            self._render_context = None


# Register the environment
def register_env():
    """Register the environment with Gymnasium."""
    from gymnasium.envs.registration import register

    register(
        id="UR5eRobotiqPickPlace-v0",
        entry_point="envs.ur5e_robotiq_pick_place_env:UR5eRobotiqPickPlaceEnv",
        max_episode_steps=500,
    )


if __name__ == "__main__":
    # Test the environment
    env = UR5eRobotiqPickPlaceEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            obs, info = env.reset()

    env.close()
