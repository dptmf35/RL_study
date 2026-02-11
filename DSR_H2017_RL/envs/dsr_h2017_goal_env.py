"""DSR H2017 GoalEnv for SAC + HER training.

This environment follows the Gymnasium-Robotics GoalEnv pattern:
- Dict observation space with {observation, achieved_goal, desired_goal}
- compute_reward(achieved_goal, desired_goal, info) method for HER
- Sparse rewards: -1 if not at goal, 0 if at goal (Fetch-style)

The objective is identical to DSRH2017AlignEnv: align the gripper directly
above a cube on a table at the approach height, with the gripper open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


__all__ = ["DSRH2017GoalEnv"]


@dataclass
class GoalAlignConfig:
    """Configuration for the goal-conditioned alignment environment."""

    cube_x_range: Tuple[float, float] = (0.6, 0.8)
    cube_y_range: Tuple[float, float] = (-0.2, 0.2)
    cube_height: float = 0.78
    approach_height_target: float = 0.10
    distance_threshold: float = 0.04  # 3D distance for success (metres)
    gripper_open_threshold: float = 0.2
    max_episode_steps: int = 200
    joint_delta_scale: float = 0.2
    frame_skip: int = 5
    randomize_home: bool = False  # Randomize robot starting pose
    home_noise_scale: float = 0.15  # Uniform noise range per joint (radians)


class DSRH2017GoalEnv(gym.Env):
    """Goal-conditioned H2017 alignment environment for SAC + HER.

    Observation Space (Dict):
        observation (16): joint_pos(6) + joint_vel(6) + gripper(1) + ee_pos(3)
        achieved_goal (3): current end-effector XYZ position
        desired_goal (3): target position (above cube centre)

    Action Space (7): joint angle deltas (6) + gripper command (1)

    Reward:
        Sparse: -1 if ||achieved - desired|| > threshold, else 0 (Fetch-style)
        Dense:  -distance
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        config: GoalAlignConfig | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.reward_type = reward_type
        self.cfg = config or GoalAlignConfig()

        # Load MuJoCo model
        asset_path = Path(__file__).parent.parent / "assets" / "h2017_2f85_merged.xml"
        if not asset_path.exists():
            raise FileNotFoundError(f"MuJoCo asset not found at {asset_path}")

        self.model = mujoco.MjModel.from_xml_path(str(asset_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.frame_skip = self.cfg.frame_skip

        # Joint metadata
        self.arm_joint_names = [f"joint_{i}" for i in range(1, 7)]
        self.gripper_joint_name = "left_driver_joint"
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.arm_joint_names
        ]
        self.gripper_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.gripper_joint_name
        )
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center"
        )
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self.cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint"
        )

        self.joint_action_scale = self.cfg.joint_delta_scale

        # Home pose (maximum vertical gripper orientation)
        self.home_qpos = np.array([0.03, -0.10, 1.51, -3.02, -1.81, 0.0])

        # Goal dimension
        self.goal_dim = 3

        # Spaces
        obs_dim = 6 + 6 + 1 + 3  # joint_pos, joint_vel, gripper, ee_pos
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

        # Episode state
        self.max_episode_steps = self.cfg.max_episode_steps
        self.current_step = 0
        self.goal = np.zeros(3, dtype=np.float32)

        # Rendering
        self.viewer = None
        self._render_context = None

    # ------------------------------------------------------------------
    # Helpers
    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

    def _sample_goal(self) -> np.ndarray:
        """Goal = position directly above cube at approach height."""
        cube_pos = self._get_cube_pos()
        goal = cube_pos.copy()
        goal[2] += self.cfg.approach_height_target
        return goal.astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    def _get_obs(self) -> Dict[str, np.ndarray]:
        joint_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.arm_joint_ids
        ])
        joint_vel = np.array([
            self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self.arm_joint_ids
        ])
        gripper_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]
        ])
        ee_pos = self._get_ee_pos()

        observation = np.concatenate([
            joint_pos, joint_vel, gripper_pos, ee_pos,
        ]).astype(np.float32)

        return {
            "observation": observation,
            "achieved_goal": ee_pos.astype(np.float32),
            "desired_goal": self.goal.copy(),
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any],
    ) -> np.ndarray:
        """HER-compatible reward (vectorised).

        Sparse: -1 if not at goal, 0 if at goal.
        Dense:  -distance.
        """
        if achieved_goal.ndim == 1:
            d = np.linalg.norm(achieved_goal - desired_goal)
        else:
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        if self.reward_type == "sparse":
            return -(d > self.cfg.distance_threshold).astype(np.float32)
        return (-10.0 * d).astype(np.float32)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        d = np.linalg.norm(achieved_goal - desired_goal)
        gripper_open = (
            self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]
            <= self.cfg.gripper_open_threshold
        )
        return bool(d < self.cfg.distance_threshold and gripper_open)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Arm home pose (with optional randomization)
        start_qpos = self.home_qpos.copy()
        if self.cfg.randomize_home:
            noise = self.np_random.uniform(
                -self.cfg.home_noise_scale, self.cfg.home_noise_scale, size=6
            )
            start_qpos += noise
            for i, jid in enumerate(self.arm_joint_ids):
                jr = self.model.jnt_range[jid]
                if np.isfinite(jr[0]) and np.isfinite(jr[1]):
                    start_qpos[i] = np.clip(start_qpos[i], jr[0], jr[1])

        for i, jid in enumerate(self.arm_joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = start_qpos[i]
            self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0

        # Open gripper
        self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]] = 0.0
        self.data.qvel[self.model.jnt_dofadr[self.gripper_joint_id]] = 0.0

        # Randomise cube
        cx = self.np_random.uniform(*self.cfg.cube_x_range)
        cy = self.np_random.uniform(*self.cfg.cube_y_range)
        cz = self.cfg.cube_height

        cq = self.model.jnt_qposadr[self.cube_joint_id]
        self.data.qpos[cq: cq + 3] = [cx, cy, cz]
        self.data.qpos[cq + 3: cq + 7] = [1.0, 0.0, 0.0, 0.0]
        cd = self.model.jnt_dofadr[self.cube_joint_id]
        self.data.qvel[cd: cd + 6] = 0.0

        self.data.ctrl[:] = 0.0
        self.current_step = 0

        mujoco.mj_forward(self.model, self.data)

        self.goal = self._sample_goal()

        obs = self._get_obs()
        info = {
            "is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"]),
            "distance": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.current_step += 1

        # Joint delta control
        joint_deltas = action[:6] * self.joint_action_scale
        current_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.arm_joint_ids
        ])
        target_pos = current_pos + joint_deltas
        for i, jid in enumerate(self.arm_joint_ids):
            jr = self.model.jnt_range[jid]
            if np.isfinite(jr[0]) and np.isfinite(jr[1]):
                target_pos[i] = np.clip(target_pos[i], jr[0], jr[1])
        self.data.ctrl[:6] = target_pos

        # Gripper: keep open for alignment (ignore learned gripper action)
        self.data.ctrl[6] = 0.0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], {}))

        is_success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        terminated = is_success
        truncated = self.current_step >= self.max_episode_steps

        info = {
            "is_success": is_success,
            "distance": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._render_context is None:
                self._render_context = mujoco.Renderer(self.model, width=640, height=480)
            self._render_context.update_scene(self.data)
            return self._render_context.render()
        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._render_context = None
