"""
Stretch Robot Grasping Environment for Reinforcement Learning

Fixed-base Stretch robot learning to grasp a cube on a table.
Uses SAC-compatible continuous action space.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


class StretchGraspEnv(gym.Env):
    """
    Stretch Robot Grasping Environment.

    The robot base is fixed, only the arm/gripper moves to grasp a cube.

    Observation Space (17 dim):
        - Lift position (1)
        - Arm extension (1) - sum of telescope joints
        - Wrist yaw position (1)
        - Gripper position (1)
        - Lift velocity (1)
        - Wrist yaw velocity (1)
        - End effector position (3)
        - Cube position (3)
        - Cube to gripper relative position (3)
        - Gripper open/close state (1)

    Action Space (4 dim, continuous [-1, 1]):
        - Lift delta (1)
        - Arm extend delta (1)
        - Wrist yaw delta (1)
        - Gripper command (1): -1=open, 1=close
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 400,  # 300 → 400 (더 느린 움직임 = 더 많은 시간)
        reward_type: str = "dense",
        randomize_cube: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.randomize_cube = randomize_cube

        # Load MuJoCo model
        model_path = Path(__file__).parent.parent / "assets" / "stretch_fixed_base.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.frame_skip = 10

        # Actuator indices (we only use lift, arm_extend, wrist_yaw, grip)
        self.ctrl_actuators = {
            "lift": 2,
            "arm_extend": 3,
            "wrist_yaw": 4,
            "grip": 5,
        }

        # Get body/site IDs
        self.gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link_gripper_slider"
        )
        # Fallback to finger if slider not found
        if self.gripper_body_id < 0:
            self.gripper_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "link_gripper_finger_left"
            )

        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self.cube_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "cube_site"
        )

        # Get gripper finger body IDs for position tracking
        self.finger_left_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link_gripper_finger_left"
        )
        self.finger_right_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link_gripper_finger_right"
        )

        # Get gripper finger geom IDs for contact detection
        # Search for fingertip geoms
        self.finger_left_geom = -1
        self.finger_right_geom = -1
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "gripper_finger_left" in name:
                self.finger_left_geom = i
            elif name and "gripper_finger_right" in name:
                self.finger_right_geom = i

        self.cube_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )

        print(f"[ENV] Gripper body ID: {self.gripper_body_id}")
        print(f"[ENV] Finger left body: {self.finger_left_body}, geom: {self.finger_left_geom}")
        print(f"[ENV] Finger right body: {self.finger_right_body}, geom: {self.finger_right_geom}")
        print(f"[ENV] Cube body: {self.cube_body_id}, geom: {self.cube_geom_id}")

        # Action space: 4 continuous actions [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space: 16 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

        # Action scaling (reduced for smoother, safer movements)
        self.action_scales = {
            "lift": 0.01,        # 0.02 → 0.01 (50% slower, safer)
            "arm_extend": 0.01,  # 0.02 → 0.01 (50% slower, safer)
            "wrist_yaw": 0.05,   # 0.1 → 0.05 (50% slower, safer)
            "grip": 0.01,        # unchanged
        }

        # Cube spawn range (on table)
        self.cube_spawn_range = {
            "x": (-0.15, 0.15),
            "y": (-0.75, -0.55),
            "z": 0.4,
        }

        # Task thresholds
        self.grasp_threshold = 0.06  # Distance to consider "close enough"
        self.lift_threshold = 0.45  # Height to consider cube lifted (0.4=table, 0.05 lift = success)

        # Episode tracking
        self.current_step = 0
        self.cube_grasped = False
        self.task_success = False

        # Control targets (for position control)
        self.ctrl_targets = np.zeros(self.model.nu)

        # Viewer
        self.viewer = None
        self._render_context = None

    def _get_gripper_pos(self) -> np.ndarray:
        """Get gripper center position (midpoint between fingers)."""
        if self.finger_left_body >= 0 and self.finger_right_body >= 0:
            left_pos = self.data.xpos[self.finger_left_body]
            right_pos = self.data.xpos[self.finger_right_body]
            return ((left_pos + right_pos) / 2).copy()
        return self.data.xpos[self.gripper_body_id].copy()

    def _get_cube_pos(self) -> np.ndarray:
        """Get cube position."""
        return self.data.xpos[self.cube_body_id].copy()

    def _get_cube_velocity(self) -> np.ndarray:
        """Get cube linear velocity."""
        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        vel_addr = self.model.jnt_dofadr[cube_joint_id]
        return self.data.qvel[vel_addr:vel_addr+3].copy()

    def _check_grasp(self) -> bool:
        """Check if cube is being grasped (contacts on both fingers)."""
        left_contact = False
        right_contact = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Check left finger contact with cube
            if (geom1 == self.finger_left_geom and geom2 == self.cube_geom_id) or \
               (geom2 == self.finger_left_geom and geom1 == self.cube_geom_id):
                left_contact = True

            # Check right finger contact with cube
            if (geom1 == self.finger_right_geom and geom2 == self.cube_geom_id) or \
               (geom2 == self.finger_right_geom and geom1 == self.cube_geom_id):
                right_contact = True

        return left_contact and right_contact

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Joint positions from sensors
        lift_pos = self.data.sensordata[0]
        arm_extension = sum(self.data.sensordata[1:5])  # Sum of telescope joints
        wrist_yaw = self.data.sensordata[5]
        gripper_pos = self.data.sensordata[6]

        # Velocities
        lift_vel = self.data.sensordata[7]
        wrist_vel = self.data.sensordata[8]

        # Positions
        ee_pos = self._get_gripper_pos()
        cube_pos = self._get_cube_pos()
        relative_pos = cube_pos - ee_pos

        # Gripper state (normalized)
        gripper_state = gripper_pos / 0.04  # Normalize to [0, 1]

        obs = np.concatenate([
            [lift_pos],           # 1
            [arm_extension],      # 1
            [wrist_yaw],          # 1
            [gripper_pos],        # 1
            [lift_vel],           # 1
            [wrist_vel],          # 1
            ee_pos,               # 3
            cube_pos,             # 3
            relative_pos,         # 3
            [gripper_state],      # 1
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        ee_pos = self._get_gripper_pos()
        cube_pos = self._get_cube_pos()

        return {
            "distance_to_cube": np.linalg.norm(ee_pos - cube_pos),
            "cube_height": cube_pos[2],
            "cube_grasped": self._check_grasp(),
            "task_success": self.task_success,
            "current_step": self.current_step,
        }

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        Improved reward with stability and approach direction.
        
        Stage 1: Reach cube (0~1)
        Stage 2: Approach from above (0~0.5) ← NEW! Prevents side collisions
        Stage 3: Grasp cube (0~2)
        Stage 4: Lift cube (0~2, only if grasped)
        Stage 5: Success (0~5)
        
        Penalties:
        - Action magnitude (-0.5) ← NEW! Prevents violent movements
        - Instability (-0.5) ← NEW! Prevents tipping
        
        Total max: 10
        """
        ee_pos = self._get_gripper_pos()
        cube_pos = self._get_cube_pos()

        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        cube_height = cube_pos[2]
        is_grasping = self._check_grasp()

        reward_components = {
            "reach": 0.0,
            "approach": 0.0,
            "grasp": 0.0,
            "lift": 0.0,
            "success": 0.0,
            "action_penalty": 0.0,
            "stability_penalty": 0.0,
        }

        if self.reward_type == "sparse":
            if cube_height > self.lift_threshold and is_grasping:
                reward_components["success"] = 10.0
                self.task_success = True
            return sum(reward_components.values()), reward_components

        # Dense reward (improved)

        # Stage 1: Reaching reward (0~1, small guidance)
        reach_reward = 1.0 - np.tanh(5.0 * dist_to_cube)
        reward_components["reach"] = reach_reward * 0.5

        # Stage 2: Approach from above (prevents side collisions!)
        # Reward for gripper being above cube
        height_diff = ee_pos[2] - cube_pos[2]  # positive = gripper above cube
        if dist_to_cube < 0.2 and not is_grasping:
            # Encourage approaching from above
            if height_diff > 0.05:  # gripper at least 5cm above
                reward_components["approach"] = 0.5
            elif height_diff < -0.05:  # gripper below cube (BAD!)
                reward_components["approach"] = -0.3

        # Stage 3: Grasping reward (0~2)
        if is_grasping:
            reward_components["grasp"] = 2.0
            self.cube_grasped = True

        # Stage 4: Lifting reward (0~2, only if grasped!)
        if is_grasping:
            lift_height = max(0, cube_height - 0.4)
            lift_reward = min(lift_height / 0.05, 1.0) * 2.0
            reward_components["lift"] = lift_reward

            # Stage 5: Success
            if cube_height > self.lift_threshold:
                reward_components["success"] = 5.0
                self.task_success = True

        # Penalty 1: Large actions (prevents violent movements)
        # Encourage smooth, small actions
        action_magnitude = np.abs(self.data.ctrl).mean()
        if action_magnitude > 0.3:  # threshold for "large" action
            reward_components["action_penalty"] = -0.5 * (action_magnitude - 0.3)

        # Penalty 2: Instability (prevents tipping)
        # Check robot base tilt (if applicable) or velocity spikes
        qvel = self.data.qvel
        velocity_magnitude = np.abs(qvel).mean()
        if velocity_magnitude > 0.5:  # threshold for "fast" movement
            reward_components["stability_penalty"] = -0.5 * (velocity_magnitude - 0.5)

        total_reward = sum(reward_components.values())
        return total_reward, reward_components

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Initialize arm to a good starting position
        # Set lift to middle position
        self.ctrl_targets[self.ctrl_actuators["lift"]] = 0.3
        # Set arm extended a bit
        self.ctrl_targets[self.ctrl_actuators["arm_extend"]] = 0.2
        # Wrist neutral
        self.ctrl_targets[self.ctrl_actuators["wrist_yaw"]] = 0.0
        # Gripper open
        self.ctrl_targets[self.ctrl_actuators["grip"]] = 0.0

        # Apply initial controls and step to settle
        self.data.ctrl[:] = self.ctrl_targets
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Randomize cube position
        if self.randomize_cube:
            cube_x = self.np_random.uniform(*self.cube_spawn_range["x"])
            cube_y = self.np_random.uniform(*self.cube_spawn_range["y"])
        else:
            cube_x, cube_y = 0.0, -0.6

        cube_z = self.cube_spawn_range["z"]

        # Set cube position
        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )
        cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1, 0, 0, 0]

        # Reset tracking
        self.current_step = 0
        self.cube_grasped = False
        self.task_success = False

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        self.current_step += 1

        # Parse and apply actions
        lift_delta = action[0] * self.action_scales["lift"]
        arm_delta = action[1] * self.action_scales["arm_extend"]
        wrist_delta = action[2] * self.action_scales["wrist_yaw"]
        grip_action = action[3]

        # Update control targets
        lift_idx = self.ctrl_actuators["lift"]
        arm_idx = self.ctrl_actuators["arm_extend"]
        wrist_idx = self.ctrl_actuators["wrist_yaw"]
        grip_idx = self.ctrl_actuators["grip"]

        # Lift
        lift_range = self.model.actuator_ctrlrange[lift_idx]
        self.ctrl_targets[lift_idx] = np.clip(
            self.ctrl_targets[lift_idx] + lift_delta,
            lift_range[0], lift_range[1]
        )

        # Arm extension
        arm_range = self.model.actuator_ctrlrange[arm_idx]
        self.ctrl_targets[arm_idx] = np.clip(
            self.ctrl_targets[arm_idx] + arm_delta,
            arm_range[0], arm_range[1]
        )

        # Wrist
        wrist_range = self.model.actuator_ctrlrange[wrist_idx]
        self.ctrl_targets[wrist_idx] = np.clip(
            self.ctrl_targets[wrist_idx] + wrist_delta,
            wrist_range[0], wrist_range[1]
        )

        # Gripper: action > 0 = close, action < 0 = open
        grip_range = self.model.actuator_ctrlrange[grip_idx]
        if grip_action > 0:
            self.ctrl_targets[grip_idx] = grip_range[1]  # Close
        else:
            self.ctrl_targets[grip_idx] = grip_range[0]  # Open

        # Disable wheels
        self.ctrl_targets[0] = 0.0  # forward
        self.ctrl_targets[1] = 0.0  # turn

        # Apply controls and step
        self.data.ctrl[:] = self.ctrl_targets
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Get outputs
        obs = self._get_obs()
        reward, reward_info = self._compute_reward()
        info = self._get_info()
        info["reward_components"] = reward_info

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
                self._render_context = mujoco.Renderer(
                    self.model, height=480, width=640
                )
            self._render_context.update_scene(self.data)
            return self._render_context.render()

        return None

    def close(self):
        """Clean up."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._render_context is not None:
            self._render_context = None


if __name__ == "__main__":
    # Test environment
    env = StretchGraspEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")

    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Episode ended at step {i+1}, reward: {reward:.2f}")
            obs, info = env.reset()

    env.close()
