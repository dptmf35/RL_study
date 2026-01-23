"""
UR5e + Robotiq 2F85 GoalEnv for SAC + HER Training

This environment follows the Gymnasium-Robotics GoalEnv pattern:
- Dict observation space with {observation, achieved_goal, desired_goal}
- compute_reward(achieved_goal, desired_goal, info) method for HER
- Sparse rewards: -1 if not at goal, 0 if at goal

This is the recommended approach for sparse-reward manipulation tasks.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class UR5eRobotiqGoalEnv(gym.Env):
    """
    GoalEnv-compatible UR5e + Robotiq environment for HER.

    Observation Space (Dict):
        - observation: robot state (joint pos, vel, gripper, etc.)
        - achieved_goal: current position of interest (EE for reach, cube for pick)
        - desired_goal: target position

    Action Space:
        - Delta joint positions (6) + gripper (1)

    Reward:
        - Sparse: -1 if ||achieved_goal - desired_goal|| > threshold, 0 otherwise
        - Dense (optional): -distance
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 100,
        reward_type: str = "sparse",  # "sparse" or "dense"
        task_mode: str = "reach",  # "reach", "pick", "pick_place"
        easy_mode: bool = False,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
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

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                         for name in self.joint_names]
        self.gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")

        # Get site/body indices
        self.pinch_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        # Goal dimension (3D position)
        self.goal_dim = 3

        # Distance thresholds for success (like Fetch)
        if self.task_mode == "reach":
            self.distance_threshold = 0.05  # 5cm for reach (Fetch uses 0.05)
        elif self.task_mode == "pick":
            self.distance_threshold = 0.05
        else:  # pick_place
            self.distance_threshold = 0.05

        # Action scaling
        self.joint_action_scale = 0.15

        # Home position
        self.home_qpos = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

        # Cube spawn range
        if self.easy_mode:
            self.cube_spawn_range = {
                "x": (0.30, 0.35),
                "y": (-0.03, 0.03),
                "z": 0.445
            }
        else:
            self.cube_spawn_range = {
                "x": (0.3, 0.45),
                "y": (-0.1, 0.1),
                "z": 0.445
            }

        # Define observation space
        # Observation: joint_pos(6) + joint_vel(6) + gripper(2) + ee_pos(3) = 17
        obs_dim = 17

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32),
        })

        # Action space: 6 joint deltas + 1 gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.goal = np.zeros(3)

        # Grasp state tracking for pick task
        self._gripper_closed = False
        self._pick_phase = 0  # 0: reach cube, 1: lift cube

        # Rendering
        self.viewer = None
        self._render_context = None

    def _get_ee_position(self) -> np.ndarray:
        """Get end effector position."""
        return self.data.site_xpos[self.pinch_site_id].copy()

    def _get_cube_position(self) -> np.ndarray:
        """Get cube position."""
        return self.data.xpos[self.cube_body_id].copy()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get observation in GoalEnv format."""
        # Robot observation
        joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]]
                             for jid in self.joint_ids])
        joint_vel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]]
                             for jid in self.joint_ids])
        gripper_pos = self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]
        gripper_vel = self.data.qvel[self.model.jnt_dofadr[self.gripper_joint_id]]
        ee_pos = self._get_ee_position()

        observation = np.concatenate([
            joint_pos,        # 6
            joint_vel,        # 6
            [gripper_pos],    # 1
            [gripper_vel],    # 1
            ee_pos,           # 3
        ]).astype(np.float32)

        # For reach and pick: achieved_goal is EE position (agent learns to move EE)
        # For pick_place: achieved_goal is cube position (agent learns to move cube)
        if self.task_mode in ["reach", "pick"]:
            achieved_goal = ee_pos.astype(np.float32)
        else:
            achieved_goal = self._get_cube_position().astype(np.float32)

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.goal.astype(np.float32),
        }

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal based on task mode."""
        if self.task_mode == "reach":
            # Goal is cube position (we want EE to reach the cube)
            return self._get_cube_position()
        elif self.task_mode == "pick":
            # Two-phase goal:
            # Phase 0: Goal is cube position (reach down to cube)
            # Phase 1: Goal is above cube (lift up)
            # Initial goal is always cube position
            return self._get_cube_position()
        else:  # pick_place
            # Goal is target location
            return self.data.site_xpos[self.target_site_id].copy()

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        Compute reward for HER.

        This method is called by HER to recompute rewards with hindsight goals.
        Must be vectorized (can handle batch of goals).

        Reward structure:
        - Sparse: -1 if not at goal, 0 if at goal (Fetch style)
        - Dense: -distance (simple, like ur5_env.py)
        """
        # Handle batch dimension
        if achieved_goal.ndim == 1:
            distance = np.linalg.norm(achieved_goal - desired_goal)
        else:
            distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if self.reward_type == "sparse":
            # Sparse reward: -1 if not at goal, 0 if at goal (Fetch style)
            return -(distance > self.distance_threshold).astype(np.float32)
        else:
            # Dense reward: simple negative distance (like ur5_env.py: -10 * distance)
            return (-10.0 * distance).astype(np.float32)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """Check if goal is achieved."""
        distance = np.linalg.norm(achieved_goal - desired_goal)

        if self.task_mode == "pick":
            # For pick: also check if cube is actually lifted (not just close to goal)
            cube_pos = self._get_cube_position()
            cube_lifted = cube_pos[2] > 0.48  # At least 4cm above table (0.44)
            return distance < self.distance_threshold and cube_lifted
        else:
            return distance < self.distance_threshold

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        obs = self._get_obs()
        achieved_goal = obs["achieved_goal"]
        desired_goal = obs["desired_goal"]

        return {
            "is_success": self._is_success(achieved_goal, desired_goal),
            "distance": np.linalg.norm(achieved_goal - desired_goal),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set robot to home position
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = self.home_qpos[i]

        # Open gripper
        gripper_qpos_addr = self.model.jnt_qposadr[self.gripper_joint_id]
        self.data.qpos[gripper_qpos_addr] = 0.0

        # Randomize cube position
        cube_x = self.np_random.uniform(*self.cube_spawn_range["x"])
        cube_y = self.np_random.uniform(*self.cube_spawn_range["y"])
        cube_z = self.cube_spawn_range["z"]

        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1, 0, 0, 0]

        # Set controls
        self.data.ctrl[:6] = self.home_qpos
        self.data.ctrl[6] = 0

        # Forward simulation
        mujoco.mj_forward(self.model, self.data)

        # Sample goal
        self.goal = self._sample_goal()

        # Reset episode tracking
        self.current_step = 0
        self._gripper_closed = False
        self._pick_phase = 0  # Start with reaching cube

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
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

        # Gripper control strategy depends on task mode
        if self.task_mode == "reach":
            # Reach: ignore gripper action, keep open
            self.data.ctrl[6] = 0
        elif self.task_mode == "pick":
            # Pick: two-phase approach
            # Phase 0: Reach cube (goal = cube position), gripper open
            # Phase 1: Lift cube (goal = above cube), gripper closed
            ee_pos = self._get_ee_position()
            cube_pos = self._get_cube_position()

            if self._pick_phase == 0:
                # Phase 0: Reaching cube
                dist_to_cube = np.linalg.norm(ee_pos - cube_pos)

                if dist_to_cube < 0.05:  # Close enough to grasp (5cm 3D distance)
                    # Close gripper and transition to lift phase
                    self.data.ctrl[6] = 0.8
                    self._gripper_closed = True
                    self._pick_phase = 1
                    # Update goal to lifted position
                    self.goal = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.10])
                else:
                    self.data.ctrl[6] = 0  # Keep open while approaching
            else:
                # Phase 1: Lifting cube - keep gripper closed
                self.data.ctrl[6] = 0.8
        else:
            # pick_place: use learned gripper action
            gripper_ctrl = (gripper_action + 1.0) * 0.4  # Map to 0-0.8
            gripper_ctrl = np.clip(gripper_ctrl, 0, 0.8)
            self.data.ctrl[6] = gripper_ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()
        info = self._get_info()

        # Compute reward using the HER-compatible method
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)

        # Check termination
        terminated = info["is_success"]
        truncated = self.current_step >= self.max_episode_steps

        # Distance penalty on episode end (truncation without success)
        if truncated and not terminated:
            distance = info["distance"]
            distance_penalty = -10.0 * min(distance, 1.0)
            reward = reward + distance_penalty
            info["distance_penalty"] = distance_penalty

        # Time bonus on success
        if terminated:
            time_bonus = 10.0 * max(0, 1.0 - (self.current_step / self.max_episode_steps))
            reward = reward + time_bonus
            info["time_bonus"] = time_bonus

        return obs, float(reward), terminated, truncated, info

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


if __name__ == "__main__":
    # Test the environment
    print("Testing UR5eRobotiqGoalEnv...")

    env = UR5eRobotiqGoalEnv(
        render_mode="human",
        task_mode="reach",
        easy_mode=True,
        reward_type="sparse",
    )

    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs keys: {obs.keys()}")
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Achieved goal: {obs['achieved_goal']}")
    print(f"Desired goal: {obs['desired_goal']}")
    print(f"Initial info: {info}")

    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.2f}, distance={info['distance']:.4f}, success={info['is_success']}")

        if terminated or truncated:
            print(f"Episode ended at step {i+1}, success={info['is_success']}")
            obs, info = env.reset()

    env.close()
