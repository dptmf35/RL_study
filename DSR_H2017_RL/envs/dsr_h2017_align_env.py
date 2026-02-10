"""DSR H2017 alignment environment for reinforcement learning.

This environment adapts the UR5e pick-and-place template to the Doosan
H2017 arm equipped with a Robotiq 2F85 gripper. The objective is to align
the gripper directly above a cube resting on a table so that the end effector
is in a grasp-ready pose. Episodes terminate when the gripper is within a
horizontal tolerance of the cube and hovers inside an approach-height band
while remaining open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


__all__ = ["DSRH2017AlignEnv", "make_env"]


@dataclass
class AlignmentConfig:
    """Configuration container for reward thresholds and spawn ranges."""

    cube_x_range: Tuple[float, float] = (0.6, 0.8)
    cube_y_range: Tuple[float, float] = (-0.2, 0.2)
    cube_height: float = 0.78  # table top (0.75) + cube half height (~0.03)
    xy_tolerance_coarse: float = 0.10
    xy_tolerance_strict: float = 0.04
    approach_height_target: float = 0.10  # target offset between ee and cube
    approach_height_band: float = 0.02
    gripper_open_threshold: float = 0.2
    max_episode_steps: int = 300
    joint_delta_scale: float = 0.2  # Increased from 0.05 for faster movement
    frame_skip: int = 5


class DSRH2017AlignEnv(gym.Env):
    """Gymnasium environment for aligning the H2017 gripper above a cube."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: AlignmentConfig | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.cfg = config or AlignmentConfig()

        asset_path = Path(__file__).parent.parent / "assets" / "h2017_2f85_merged.xml"
        if not asset_path.exists():
            raise FileNotFoundError(f"MuJoCo asset not found at {asset_path}")

        self.model = mujoco.MjModel.from_xml_path(str(asset_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.frame_skip = self.cfg.frame_skip

        # Joint/actuator metadata
        self.arm_joint_names = [f"joint_{i}" for i in range(1, 7)]
        self.gripper_joint_name = "left_driver_joint"
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.arm_joint_names
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
        self.goal_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "alignment_site"
        )

        # Action/observation setup
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.joint_action_scale = self.cfg.joint_delta_scale

        obs_size = 6 + 6 + 1 + 3 + 3 + 3 + 3 + 2 + 1  # joints, vels, gripper, ee, cube, vel, relative, direction_xy, distance_xy
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Episode tracking
        self.max_episode_steps = self.cfg.max_episode_steps
        self.current_step = 0
        self.task_success = False
        self.success_step = None  # Track when success was first achieved

        # Rendering state
        self.viewer = None
        self._render_context = None

        # Home joint configuration - H2017 specific kinematics
        # Maximum vertical gripper + accurate position
        # Result: ee at (0.662, 0.173, 0.867), downward align 0.805
        self.home_qpos = np.array([
             0.0300,   # Base
            -0.1000,   # Shoulder
             1.5100,   # Elbow
            -3.0200,   # Wrist1
            -1.8100,   # Wrist2
             0.0000,   # Wrist3
        ])

    # ------------------------------------------------------------------
    # Gymnasium API helpers
    def _get_obs(self) -> np.ndarray:
        joint_pos = np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.arm_joint_ids]
        )
        joint_vel = np.array(
            [self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self.arm_joint_ids]
        )
        gripper_pos = np.array([
            self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]
        ])

        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        cube_pos = self.data.xpos[self.cube_body_id].copy()

        cube_joint_addr = self.model.jnt_dofadr[self.cube_joint_id]
        cube_vel = self.data.qvel[cube_joint_addr : cube_joint_addr + 3].copy()

        relative = cube_pos - ee_pos
        relative_xy = cube_pos[:2] - ee_pos[:2]  # XY direction to cube
        distance_xy = np.linalg.norm(relative_xy)

        # Normalized direction to cube in XY plane
        direction_xy = relative_xy / (distance_xy + 1e-6)

        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                gripper_pos,
                ee_pos,
                cube_pos,
                cube_vel,
                relative,
                direction_xy,  # Added: normalized XY direction to cube
                [distance_xy],  # Added: XY distance
            ]
        ).astype(np.float32)
        return obs

    def _get_info(self) -> Dict[str, float]:
        ee_pos = self.data.site_xpos[self.ee_site_id]
        cube_pos = self.data.xpos[self.cube_body_id]

        distance_xy = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        height_difference = ee_pos[2] - cube_pos[2]
        goal_pos = self.data.site_xpos[self.goal_site_id]
        goal_distance = np.linalg.norm(ee_pos - goal_pos)

        return {
            "distance_xy": float(distance_xy),
            "height_error": float(height_difference - self.cfg.approach_height_target),
            "goal_distance": float(goal_distance),
            "task_success": bool(self.task_success),
            "current_step": self.current_step,
        }

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        gripper_opening = self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]]

        distance_xy = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
        height_error = ee_pos[2] - cube_pos[2] - self.cfg.approach_height_target

        reward_components: Dict[str, float] = {
            "distance": -5.0 * distance_xy,  # XY alignment penalty
            "height": -5.0 * abs(height_error),  # Height penalty - INCREASED to prevent reward hacking
            "coarse_align_bonus": 0.0,
            "tight_align_bonus": 0.0,
            "success": 0.0,
            "time": -0.01,
        }

        # Bonuses only given when BOTH XY and height are aligned
        # This prevents agent from exploiting XY alignment while ignoring height
        aligned_height = abs(height_error) <= self.cfg.approach_height_band  # ±2cm

        # NO bonuses given - only dense rewards
        # This ensures agent must truly align both XY and height to minimize penalties
        # Bonuses were causing reward hacking where agent ignores height

        gripper_open = gripper_opening <= self.cfg.gripper_open_threshold

        if (
            distance_xy < self.cfg.xy_tolerance_strict
            and aligned_height
            and gripper_open
        ):
            # Large success reward to incentivize reaching goal
            reward_components["success"] = 100.0  # Increased from 10.0

            if not self.task_success:
                # First time achieving success
                self.task_success = True
                self.success_step = self.current_step
            else:
                # Already successful - give bonus for staying aligned
                reward_components["success"] = 10.0  # Smaller bonus for maintaining

        total_reward = sum(reward_components.values())
        return total_reward, reward_components

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Set arm to home configuration
        for i, jid in enumerate(self.arm_joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = self.home_qpos[i]
            self.data.qvel[self.model.jnt_dofadr[jid]] = 0.0

        # Open gripper
        self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_id]] = 0.0
        self.data.qvel[self.model.jnt_dofadr[self.gripper_joint_id]] = 0.0

        # Randomize cube placement
        cube_x = self.np_random.uniform(*self.cfg.cube_x_range)
        cube_y = self.np_random.uniform(*self.cfg.cube_y_range)
        cube_z = self.cfg.cube_height

        cube_qpos_addr = self.model.jnt_qposadr[self.cube_joint_id]
        self.data.qpos[cube_qpos_addr : cube_qpos_addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_addr + 3 : cube_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]

        cube_joint_addr = self.model.jnt_dofadr[self.cube_joint_id]
        self.data.qvel[cube_joint_addr : cube_joint_addr + 6] = 0.0

        # Clear actuator controls
        self.data.ctrl[:] = 0.0

        self.current_step = 0
        self.task_success = False
        self.success_step = None

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.current_step += 1

        joint_deltas = action[:6] * self.joint_action_scale
        gripper_cmd = action[6]

        current_joint_pos = np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.arm_joint_ids]
        )
        target_joint_pos = current_joint_pos + joint_deltas

        for i, jid in enumerate(self.arm_joint_ids):
            jrange = self.model.jnt_range[jid]
            if np.isfinite(jrange[0]) and np.isfinite(jrange[1]):
                target_joint_pos[i] = np.clip(target_joint_pos[i], jrange[0], jrange[1])

        # Set control targets (actuators are ordered identically to joints)
        self.data.ctrl[:6] = target_joint_pos

        # Gripper command: positive closes, negative opens
        if gripper_cmd > 0:
            self.data.ctrl[6] = 0.8  # close toward upper limit
        else:
            self.data.ctrl[6] = 0.0  # open

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, reward_info = self._compute_reward()
        info = self._get_info()
        info["reward_components"] = reward_info

        # Terminate episode immediately upon success (or after holding for a few steps)
        terminated = False
        if self.task_success and self.success_step is not None:
            steps_since_success = self.current_step - self.success_step
            if steps_since_success >= 5:  # Hold success for 5 steps, then terminate
                terminated = True

        truncated = self.current_step >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering utilities
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    import mujoco.viewer  # type: ignore
                except ImportError as exc:
                    raise RuntimeError(
                        "mujoco.viewer is not available. Install mujoco>=3.1.1 with viewer support."
                    ) from exc
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


def make_env(**kwargs) -> DSRH2017AlignEnv:
    """Factory helper mirroring existing RL_study scripts."""

    return DSRH2017AlignEnv(**kwargs)
