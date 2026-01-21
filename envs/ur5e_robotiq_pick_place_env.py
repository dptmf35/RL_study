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

        # Action scaling
        self.joint_action_scale = 0.1
        self.gripper_action_threshold = 0.0  # Threshold for gripper open/close

        # Home position
        self.home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

        # Workspace limits for cube spawning (relative to table at 0.5, 0, 0.4)
        if self.easy_mode:
            self.cube_spawn_range = {
                "x": (0.45, 0.55),
                "y": (-0.05, 0.05),
                "z": 0.435  # On table surface
            }
        else:
            self.cube_spawn_range = {
                "x": (0.4, 0.6),
                "y": (-0.15, 0.15),
                "z": 0.435
            }
        
        self.target_position = np.array([0.5, 0.2, 0.42])

        # Task thresholds
        self.grasp_height_threshold = 0.45  # Height to consider cube grasped
        self.place_threshold = 0.05
        self.reach_threshold = 0.05

        # Episode tracking
        self.current_step = 0
        self.cube_grasped = False
        self.task_success = False
        self.gripper_was_open_near_cube = False

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

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        ee_pos = self.data.site_xpos[self.pinch_site_id]
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
        ee_pos = self.data.site_xpos[self.pinch_site_id].copy()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        target_pos = self.data.site_xpos[self.target_site_id].copy()

        # Get gripper state (0 = open, 0.8 = closed for Robotiq)
        gripper_pos = self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]
        gripper_is_open = gripper_pos < 0.1
        gripper_is_closed = gripper_pos > 0.5

        # Distances
        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_to_target = np.linalg.norm(cube_pos - target_pos)
        cube_height = cube_pos[2]
        table_height = 0.42

        reward_components = {
            "reach": 0.0,
            "gripper_prep": 0.0,
            "grasp": 0.0,
            "lift": 0.0,
            "place": 0.0,
            "success": 0.0,
        }

        if self.reward_type == "sparse":
            if dist_to_target < self.place_threshold and cube_height > table_height:
                reward_components["success"] = 100.0
                self.task_success = True
            return sum(reward_components.values()), reward_components

        # Dense reward
        # Stage 1: Reaching
        if dist_to_cube < 0.30:
            reward_components["reach"] = 0.5
        if dist_to_cube < 0.15:
            reward_components["reach"] = 1.0
        if dist_to_cube < 0.08:
            reward_components["reach"] = 1.5
        if dist_to_cube < 0.05:
            reward_components["reach"] = 2.0

        # Stage 2: Gripper preparation
        if dist_to_cube < 0.10:
            if gripper_is_open:
                self.gripper_was_open_near_cube = True
                reward_components["gripper_prep"] = 3.0
            elif not self.gripper_was_open_near_cube:
                reward_components["gripper_prep"] = -1.0

        # Stage 3: Grasping
        if dist_to_cube < 0.06:
            if gripper_is_closed:
                if self.gripper_was_open_near_cube:
                    reward_components["grasp"] = 5.0
                    
                    if cube_height > table_height + 0.02:
                        self.cube_grasped = True
                        reward_components["grasp"] = 8.0
                else:
                    reward_components["grasp"] = -0.5

        # Stage 4: Lifting
        if self.cube_grasped:
            lift_height = max(0, cube_height - table_height)
            lift_reward = min(lift_height * 30.0, 10.0)
            reward_components["lift"] = lift_reward

            # Stage 5: Placement
            if cube_height > table_height + 0.03:
                place_reward = (1.0 - np.tanh(3.0 * dist_to_target)) * 5.0
                reward_components["place"] = place_reward

        # Stage 6: Success
        if self.task_mode == "reach":
            if dist_to_cube < 0.05:
                reward_components["success"] = 20.0
                self.task_success = True
        elif self.task_mode == "pick":
            if self.cube_grasped and cube_height > table_height + 0.05:
                reward_components["success"] = 50.0
                self.task_success = True
        else:  # pick_place
            if dist_to_target < self.place_threshold and cube_height > table_height:
                reward_components["success"] = 100.0
                self.task_success = True

        # Time penalty
        time_penalty = -0.02

        total_reward = sum(reward_components.values()) + time_penalty
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
            cube_x, cube_y, cube_z = 0.5, 0.0, 0.435

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
        self.cube_grasped = False
        self.task_success = False
        self.gripper_was_open_near_cube = False

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

        # Gripper control: map [-1, 1] to [0, 255]
        # -1 = open (0), 1 = close (255)
        gripper_ctrl = (gripper_action + 1.0) * 127.5  # Map to 0-255
        gripper_ctrl = np.clip(gripper_ctrl, 0, 255)
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
