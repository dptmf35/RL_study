"""
Go2 Locomotion Environment with Box-Based Terrain.

Extends Go2Env with:
- Box primitive terrain (real physics collision, not visual-only heightfield)
- Terrain-relative rewards and termination
- Curriculum learning via runtime terrain height scaling
- Same 45-dim observation space as flat env (transfer learning compatible)
"""

import numpy as np
import mujoco

from envs.go2_env import Go2Env, NUM_JOINTS
from envs.terrain import TerrainGenerator


class Go2TerrainEnv(Go2Env):
    """Go2 environment with box-based terrain.

    Terrain is built from stacked box primitives (stepped-pyramid mounds)
    which provide reliable physics collision with Go2's small foot spheres.

    Observation: same 45-dim as Go2Env (transfer learning compatible).
    Physics: ground plane + box terrain geoms (real collision).
    Rewards: terrain-relative base height target.
    Curriculum: scales terrain geom heights from flat to full difficulty.
    """

    def __init__(
        self,
        difficulty: float = 0.5,
        terrain_seed: int | None = None,
        curriculum: bool = False,
        curriculum_start: float = 0.0,
        curriculum_end: float | None = None,
        curriculum_steps: int = 1_000_000,
        **kwargs,
    ):
        # Target difficulty (terrain generated at this level)
        self._target_difficulty = curriculum_end or difficulty

        # Generate terrain and scene XML
        self.terrain_gen = TerrainGenerator()
        self._terrain_seed = terrain_seed
        self.terrain_gen.generate(
            difficulty=self._target_difficulty, seed=terrain_seed
        )
        scene_xml = self.terrain_gen.generate_scene_xml()

        # Curriculum setup
        self._curriculum = curriculum
        self._curriculum_start = curriculum_start
        self._curriculum_end = self._target_difficulty
        self._curriculum_steps = curriculum_steps
        self._curriculum_step_count = 0
        self._difficulty_scale = curriculum_start if curriculum else 1.0

        # Initialize parent with terrain scene XML
        kwargs.pop("scene_xml_path", None)
        super().__init__(scene_xml_path=scene_xml, **kwargs)

        # Find terrain geom IDs for curriculum scaling
        self._terrain_geom_ids = []
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.startswith("terrain_"):
                self._terrain_geom_ids.append(i)
        self._terrain_geom_ids = np.array(self._terrain_geom_ids, dtype=np.int32)

        # Store reference z-positions and z-sizes for curriculum scaling
        self._terrain_ref_pos_z, self._terrain_ref_size_z = (
            self.terrain_gen.get_terrain_geom_ref_data()
        )

        # Apply initial difficulty scale
        self._apply_difficulty_scale(self._difficulty_scale)

    def _apply_difficulty_scale(self, scale):
        """Scale terrain box heights by difficulty factor.

        At scale=0: all terrain boxes are flat (embedded in ground plane).
        At scale=1: full terrain height.
        """
        self._difficulty_scale = scale
        min_size = 0.0005  # minimum half-height to avoid degenerate boxes
        for i, gid in enumerate(self._terrain_geom_ids):
            self.model.geom_pos[gid, 2] = self._terrain_ref_pos_z[i] * scale
            self.model.geom_size[gid, 2] = max(
                self._terrain_ref_size_z[i] * scale, min_size
            )

    def _get_terrain_height_at(self, x, y):
        """Get terrain height at world coordinates (scaled by difficulty)."""
        return self.terrain_gen.sample_height(x, y, self._difficulty_scale)

    def _get_feet_contact(self) -> np.ndarray:
        """Check which feet are in contact with ground or terrain.

        Unlike flat env (checks floor geom only), this checks contacts
        against any worldbody geom (body_id=0), including terrain boxes.
        """
        contacts = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for foot_idx, geom_id in enumerate(self._foot_geom_ids):
                if c.geom1 == geom_id or c.geom2 == geom_id:
                    other = c.geom2 if c.geom1 == geom_id else c.geom1
                    # Contact with any worldbody geom (floor + terrain boxes)
                    if self.model.geom_bodyid[other] == 0:
                        contacts[foot_idx] = True
        return contacts

    def _compute_observation(self) -> np.ndarray:
        return super()._compute_observation()  # 45 dims (same as flat env)

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
        # Update curriculum difficulty
        if self._curriculum:
            self._curriculum_step_count += self._step_count
            progress = min(self._curriculum_step_count / self._curriculum_steps, 1.0)
            new_scale = (
                self._curriculum_start
                + (self._curriculum_end - self._curriculum_start) * progress
            )
            if abs(new_scale - self._difficulty_scale) > 0.01:
                self._apply_difficulty_scale(new_scale)

        obs, info = super().reset(seed=seed, options=options)

        # Adjust base height for terrain at spawn position
        base_x, base_y = self.data.qpos[0], self.data.qpos[1]
        terrain_z = self._get_terrain_height_at(base_x, base_y)
        self.data.qpos[2] = terrain_z + 0.27

        mujoco.mj_forward(self.model, self.data)

        # Recompute observation with correct height
        obs = self._compute_observation()
        info["terrain_difficulty"] = self._difficulty_scale

        return obs, info
