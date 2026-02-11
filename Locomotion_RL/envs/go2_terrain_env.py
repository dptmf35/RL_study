"""
Go2 Locomotion Environment with Terrain Support.

Extends Go2Env with:
- Procedural heightfield terrain (slopes, stairs, rough)
- Terrain-aware observations (foot heights + forward scan)
- Terrain-relative rewards
"""

import numpy as np
import mujoco
from gymnasium import spaces

from envs.go2_env import Go2Env, NUM_JOINTS
from envs.terrain import TerrainGenerator, ensure_terrain_scene_xml


class Go2TerrainEnv(Go2Env):
    """Go2 environment with heightfield terrain.

    Observation (53-dim):
        Base observations from Go2Env: 45
        + Terrain height at 4 foot positions (relative to base): 4
        + Forward terrain scan (4 points ahead): 4
        Total: 53

    The terrain is generated procedurally at environment creation.
    Difficulty controls terrain feature height (0=flat, 1=hard).
    """

    def __init__(
        self,
        difficulty: float = 0.5,
        terrain_seed: int | None = None,
        **kwargs,
    ):
        # Ensure terrain scene XML exists
        scene_xml = ensure_terrain_scene_xml()

        # Generate terrain heightfield
        self.terrain_gen = TerrainGenerator()
        self.terrain_heights = self.terrain_gen.generate(
            difficulty=difficulty, seed=terrain_seed
        )
        self.terrain_difficulty = difficulty

        # Initialize parent with terrain scene XML
        kwargs.pop("scene_xml_path", None)
        super().__init__(scene_xml_path=scene_xml, **kwargs)

        # Inject heightfield data into the loaded model
        hfield_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain"
        )
        adr = self.model.hfield_adr[hfield_id]
        n = self.terrain_gen.nrow * self.terrain_gen.ncol
        self.model.hfield_data[adr : adr + n] = self.terrain_heights.flatten()

        # Extend observation space for terrain info
        self._terrain_obs_dim = 8  # 4 foot heights + 4 forward scan
        obs_dim = 45 + self._terrain_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_terrain_height_at(self, x, y):
        """Get terrain height at world coordinates."""
        return self.terrain_gen.sample_height(self.terrain_heights, x, y)

    def _compute_terrain_obs(self) -> np.ndarray:
        """Compute terrain-specific observations.

        Returns:
            8-dim array:
                [0:4] terrain height at each foot relative to base z (scaled)
                [4:8] forward scan heights relative to base z (scaled)
        """
        base_pos = self._get_base_pos()
        base_z = base_pos[2]

        # Get body-frame forward direction (projected to xy)
        quat = self._get_base_quat()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        forward_xy = rot[:2, 0]  # first column of rotation matrix (x-axis in body)

        # Terrain height at each foot position
        foot_terrain = np.zeros(4)
        for i, geom_id in enumerate(self._foot_geom_ids):
            foot_pos = self.data.geom_xpos[geom_id]
            terrain_z = self._get_terrain_height_at(foot_pos[0], foot_pos[1])
            foot_terrain[i] = (terrain_z - base_z) * 5.0  # scaled

        # Forward terrain scan: 4 points at 0.15, 0.30, 0.45, 0.60m ahead
        scan_heights = np.zeros(4)
        for i, dist in enumerate([0.15, 0.30, 0.45, 0.60]):
            scan_x = base_pos[0] + forward_xy[0] * dist
            scan_y = base_pos[1] + forward_xy[1] * dist
            terrain_z = self._get_terrain_height_at(scan_x, scan_y)
            scan_heights[i] = (terrain_z - base_z) * 5.0  # scaled

        return np.concatenate([foot_terrain, scan_heights])

    def _compute_observation(self) -> np.ndarray:
        base_obs = super()._compute_observation()  # 45 dims
        terrain_obs = self._compute_terrain_obs()   # 8 dims
        return np.concatenate([base_obs, terrain_obs]).astype(np.float32)

    def _compute_reward(self) -> tuple[float, dict]:
        """Compute reward with terrain-relative base height."""
        info = {}
        total_reward = 0.0

        base_lin_vel = self._get_base_lin_vel()
        base_ang_vel = self._get_base_ang_vel()
        joint_vel = self._get_joint_vel()
        joint_acc = (joint_vel - self._last_joint_vel) / self.control_dt
        proj_gravity = self._get_projected_gravity()
        base_pos = self._get_base_pos()
        base_height = base_pos[2]
        contacts = self._get_feet_contact()

        # Terrain height under the robot
        terrain_z = self._get_terrain_height_at(base_pos[0], base_pos[1])

        # Forward velocity tracking
        vx_error = self._command[0] - base_lin_vel[0]
        r_forward = np.exp(-4.0 * vx_error ** 2)
        total_reward += self.rew_scales["forward_vel"] * r_forward
        info["r_forward"] = r_forward

        # Lateral velocity tracking
        vy_error = self._command[1] - base_lin_vel[1]
        r_lateral = np.exp(-4.0 * vy_error ** 2)
        total_reward += self.rew_scales["lateral_vel"] * r_lateral
        info["r_lateral"] = r_lateral

        # Yaw rate tracking
        yaw_error = self._command[2] - base_ang_vel[2]
        r_yaw = np.exp(-4.0 * yaw_error ** 2)
        total_reward += self.rew_scales["yaw_rate"] * r_yaw
        info["r_yaw"] = r_yaw

        # Torque penalty
        torques = self.data.ctrl
        r_torque = np.sum(torques ** 2)
        total_reward += self.rew_scales["torque"] * r_torque
        info["r_torque"] = r_torque

        # Joint acceleration penalty
        r_joint_acc = np.sum(joint_acc ** 2)
        total_reward += self.rew_scales["joint_acc"] * r_joint_acc
        info["r_joint_acc"] = r_joint_acc

        # Action rate penalty
        r_action_rate = np.sum(
            (self._last_action - self.data.ctrl / self._torque_limits.max()) ** 2
        )
        total_reward += self.rew_scales["action_rate"] * r_action_rate
        info["r_action_rate"] = r_action_rate

        # Orientation penalty
        r_orientation = np.sum(proj_gravity[:2] ** 2)
        total_reward += self.rew_scales["orientation"] * r_orientation
        info["r_orientation"] = r_orientation

        # Base height penalty (TERRAIN-RELATIVE)
        target_height = terrain_z + 0.27
        r_height = (base_height - target_height) ** 2
        total_reward += self.rew_scales["base_height"] * r_height
        info["r_height"] = r_height

        # Feet air time reward
        self._feet_air_time += self.control_dt
        self._feet_air_time *= ~contacts
        r_air_time = np.sum((self._feet_air_time - 0.5) * contacts)
        r_air_time = np.clip(r_air_time, 0.0, None)
        total_reward += self.rew_scales["feet_air_time"] * r_air_time
        info["r_air_time"] = r_air_time

        # Alive bonus
        total_reward += self.rew_scales["alive"]

        info["total_reward"] = total_reward
        return total_reward, info

    def _check_termination(self) -> tuple[bool, bool]:
        """Terrain-relative termination check."""
        proj_gravity = self._get_projected_gravity()
        base_pos = self._get_base_pos()
        terrain_z = self._get_terrain_height_at(base_pos[0], base_pos[1])

        # Body tilt too large
        body_tilt = np.arccos(np.clip(-proj_gravity[2], -1.0, 1.0))
        terminated = bool(body_tilt > self.max_body_tilt)

        # Base too close to terrain surface
        height_above_terrain = base_pos[2] - terrain_z
        terminated = terminated or bool(height_above_terrain < 0.10)

        # Truncated by time limit
        truncated = self._step_count >= self.max_episode_steps

        return terminated, truncated

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Adjust base height for terrain at spawn position
        base_x, base_y = self.data.qpos[0], self.data.qpos[1]
        terrain_z = self._get_terrain_height_at(base_x, base_y)
        self.data.qpos[2] = terrain_z + 0.27

        mujoco.mj_forward(self.model, self.data)

        # Recompute observation with correct height
        obs = self._compute_observation()
        info["terrain_difficulty"] = self.terrain_difficulty

        return obs, info
