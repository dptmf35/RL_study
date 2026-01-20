"""
UR5e Pick and Place Environment for Reinforcement Learning

This environment simulates a UR5e robot arm performing pick and place tasks
with a cube object. The cube's initial position is randomized, and the robot
must learn to pick it up and place it at a target location.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class UR5ePickPlaceEnv(gym.Env):
    """
    UR5e Pick and Place Environment.

    Observation Space:
        - Joint positions (6)
        - Joint velocities (6)
        - Gripper position (2)
        - End effector position (3)
        - Cube position (3)
        - Cube velocity (3)
        - Target position (3)
        - Relative cube to gripper (3)
        - Relative cube to target (3)
        Total: 32 dimensions

    Action Space:
        - Delta joint positions (6) - continuous
        - Gripper command (1) - continuous (0: open, 1: close)
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
    ):
        """
        Initialize the UR5e Pick and Place environment.

        Args:
            render_mode: "human" for live rendering, "rgb_array" for image output
            max_episode_steps: Maximum steps per episode
            reward_type: "dense" or "sparse" reward
            randomize_cube: Whether to randomize cube initial position
            randomize_target: Whether to randomize target position
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.randomize_cube = randomize_cube
        self.randomize_target = randomize_target

        # Load MuJoCo model
        model_path = Path(__file__).parent.parent / "assets" / "ur5e_pick_place.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.dt = self.model.opt.timestep
        self.frame_skip = 10  # Number of simulation steps per action

        # Robot configuration
        self.n_joints = 6
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.gripper_joint_names = ["finger_left_joint", "finger_right_joint"]

        # Get joint indices
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                         for name in self.joint_names]
        self.gripper_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                  for name in self.gripper_joint_names]

        # Get actuator indices
        self.actuator_ids = list(range(6))  # First 6 actuators are for arm
        self.gripper_actuator_ids = [6, 7]  # Last 2 are for gripper

        # Get site/body indices
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        # Action and observation spaces
        # Actions: delta joint positions (6) + gripper (1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Observations: see docstring
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32
        )

        # Action scaling
        self.joint_action_scale = 0.1  # Scale for delta joint actions
        self.gripper_action_threshold = 0.5  # Threshold for gripper open/close

        # Home position for robot
        self.home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

        # Workspace limits for randomization
        self.cube_spawn_range = {
            "x": (0.3, 0.6),
            "y": (-0.3, 0.1),
            "z": 0.475  # Fixed height (on table)
        }
        self.target_position = np.array([0.6, 0.2, 0.475])  # Default target

        # Task thresholds
        self.grasp_height_threshold = 0.52  # Height to consider cube grasped
        self.place_threshold = 0.05  # Distance threshold for successful placement
        self.reach_threshold = 0.05  # Distance threshold for reaching cube

        # Episode tracking
        self.current_step = 0
        self.cube_grasped = False
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

        # Gripper position
        gripper_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                               for jid in self.gripper_joint_ids])

        # End effector position
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()

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
            gripper_pos,      # 2
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
        # Find the cube's qvel index (freejoint has 6 DOFs: 3 position + 3 orientation)
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        vel_addr = self.model.jnt_dofadr[cube_joint_id]
        return self.data.qvel[vel_addr:vel_addr+3].copy()

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.xpos[self.cube_body_id]
        target_pos = self.data.site_xpos[self.target_site_id]

        return {
            "distance_to_cube": np.linalg.norm(ee_pos - cube_pos),
            "distance_to_target": np.linalg.norm(cube_pos - target_pos),
            "cube_height": cube_pos[2],
            "cube_grasped": self.cube_grasped,
            "task_success": self.task_success,
            "current_step": self.current_step,
        }

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """Compute reward based on current state."""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.xpos[self.cube_body_id]
        target_pos = self.data.site_xpos[self.target_site_id]

        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_to_target = np.linalg.norm(cube_pos - target_pos)
        cube_height = cube_pos[2]

        reward_components = {
            "reach": 0.0,
            "grasp": 0.0,
            "lift": 0.0,
            "place": 0.0,
            "success": 0.0,
        }

        if self.reward_type == "sparse":
            # Sparse reward: only reward task completion
            if dist_to_target < self.place_threshold and cube_height > 0.45:
                reward_components["success"] = 100.0
                self.task_success = True
            return sum(reward_components.values()), reward_components

        # Dense reward
        # 1. Reaching reward (encourage getting close to cube)
        reach_reward = -dist_to_cube
        reward_components["reach"] = reach_reward * 1.0

        # 2. Grasping reward (encourage grasping when close)
        if dist_to_cube < self.reach_threshold:
            # Check if gripper is closed and cube is between fingers
            gripper_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                                   for jid in self.gripper_joint_ids])
            if np.mean(gripper_pos) > 0.01:  # Gripper somewhat closed
                reward_components["grasp"] = 2.0

                # Check if cube is lifted
                if cube_height > self.grasp_height_threshold:
                    self.cube_grasped = True
                    reward_components["lift"] = 5.0

        # 3. Placement reward (encourage moving cube to target)
        if self.cube_grasped or cube_height > self.grasp_height_threshold:
            place_reward = -dist_to_target
            reward_components["place"] = place_reward * 2.0

            # Task completion
            if dist_to_target < self.place_threshold:
                reward_components["success"] = 100.0
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

        # Open gripper
        for jid in self.gripper_joint_ids:
            self.data.qpos[self.model.jnt_qposadr[jid]] = 0.0

        # Randomize cube position
        if self.randomize_cube:
            cube_x = self.np_random.uniform(*self.cube_spawn_range["x"])
            cube_y = self.np_random.uniform(*self.cube_spawn_range["y"])
            cube_z = self.cube_spawn_range["z"]
        else:
            cube_x, cube_y, cube_z = 0.4, -0.2, 0.475

        # Set cube position (freejoint: 3 pos + 4 quat)
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1, 0, 0, 0]  # Identity quaternion

        # Randomize target if needed
        if self.randomize_target:
            target_x = self.np_random.uniform(0.4, 0.7)
            target_y = self.np_random.uniform(-0.1, 0.3)
            # Update target site position (need to modify model)
            self.target_position = np.array([target_x, target_y, 0.475])

        # Reset episode tracking
        self.current_step = 0
        self.cube_grasped = False
        self.task_success = False

        # Forward simulation to update state
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

        # Gripper control
        if gripper_action > self.gripper_action_threshold:
            # Close gripper
            self.data.ctrl[6] = 0.04
            self.data.ctrl[7] = 0.04
        else:
            # Open gripper
            self.data.ctrl[6] = 0.0
            self.data.ctrl[7] = 0.0

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


# Register the environment with Gymnasium
def register_env():
    """Register the environment with Gymnasium."""
    from gymnasium.envs.registration import register

    register(
        id="UR5ePickPlace-v0",
        entry_point="envs.ur5e_pick_place_env:UR5ePickPlaceEnv",
        max_episode_steps=500,
    )


if __name__ == "__main__":
    # Test the environment
    env = UR5ePickPlaceEnv(render_mode="human")
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
