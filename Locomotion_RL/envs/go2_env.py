"""
Unitree Go2 Locomotion Environment for MuJoCo.

Custom Gymnasium environment for training quadruped locomotion policies.
Uses position-based PD control with torque limits from the Go2 actuator specs.
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


# Path to Go2 scene XML (relative to this file)
_MENAGERIE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "mujoco_menagerie", "unitree_go2"
)
DEFAULT_SCENE_XML = os.path.join(_MENAGERIE_DIR, "scene.xml")

# Joint ordering: FL(hip,thigh,calf), FR(...), RL(...), RR(...)
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

FOOT_GEOM_NAMES = ["FL", "FR", "RL", "RR"]

# Default standing joint angles (from keyframe "home")
DEFAULT_JOINT_ANGLES = np.array([
    0.0, 0.9, -1.8,   # FL
    0.0, 0.9, -1.8,   # FR
    0.0, 0.9, -1.8,   # RL
    0.0, 0.9, -1.8,   # RR
])

NUM_JOINTS = 12


class Go2Env(gym.Env):
    """Unitree Go2 quadruped locomotion environment.

    Observation (46-dim):
        - Base linear velocity (body frame)    : 3
        - Base angular velocity (body frame)   : 3
        - Projected gravity vector (body frame): 3
        - Command velocity (vx, vy, yaw_rate)  : 3
        - Joint positions (relative to default): 12
        - Joint velocities                     : 12
        - Previous actions                     : 12
        minus base_lin_vel (not directly observable in real robot),
        => we remove it for sim-to-real friendliness

    Actually we keep it simple and observable for simulation training:
        - projected gravity    : 3
        - command velocity     : 3
        - joint pos (relative) : 12
        - joint vel            : 12
        - previous actions     : 12
        - base angular vel     : 3
        Total: 45

    Action (12-dim):
        Target joint position offsets from default stance, scaled by action_scale.

    Reward:
        Multi-term reward encouraging forward locomotion with smooth gait.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scene_xml_path: str = DEFAULT_SCENE_XML,
        render_mode: str | None = None,
        # Simulation
        sim_dt: float = 0.005,         # 200 Hz physics
        control_dt: float = 0.02,      # 50 Hz control
        max_episode_steps: int = 1000, # 20s episodes
        # PD control gains
        kp: float = 40.0,
        kd: float = 1.0,
        action_scale: float = 0.25,
        # Commands
        cmd_vx_range: tuple = (0.0, 1.0),
        cmd_vy_range: tuple = (-0.3, 0.3),
        cmd_yaw_range: tuple = (-0.5, 0.5),
        # Reward scales
        rew_forward_vel: float = 1.0,
        rew_lateral_vel: float = 0.5,
        rew_yaw_rate: float = 0.5,
        rew_torque: float = -1e-5,
        rew_joint_acc: float = -2.5e-7,
        rew_action_rate: float = -0.01,
        rew_orientation: float = -1.0,
        rew_base_height: float = -10.0,
        rew_feet_air_time: float = 1.0,
        rew_alive: float = 0.5,
        # Termination
        max_body_tilt: float = 0.8,    # rad, terminate if tilted too much
        min_base_height: float = 0.15,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)

        # Timing
        self.model.opt.timestep = sim_dt
        self.sim_dt = sim_dt
        self.control_dt = control_dt
        self.n_substeps = int(control_dt / sim_dt)
        self.max_episode_steps = max_episode_steps

        # PD control
        self.kp = kp
        self.kd = kd
        self.action_scale = action_scale
        self.default_joint_angles = DEFAULT_JOINT_ANGLES.copy()

        # Cache joint and actuator IDs
        self._joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ])
        self._joint_qpos_adr = np.array([
            self.model.jnt_qposadr[jid] for jid in self._joint_ids
        ])
        self._joint_dof_adr = np.array([
            self.model.jnt_dofadr[jid] for jid in self._joint_ids
        ])
        self._foot_geom_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in FOOT_GEOM_NAMES
        ])
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        # Torque limits from actuator model
        self._torque_limits = self.model.actuator_ctrlrange[:, 1].copy()

        # Command ranges
        self.cmd_vx_range = cmd_vx_range
        self.cmd_vy_range = cmd_vy_range
        self.cmd_yaw_range = cmd_yaw_range

        # Reward scales
        self.rew_scales = {
            "forward_vel": rew_forward_vel,
            "lateral_vel": rew_lateral_vel,
            "yaw_rate": rew_yaw_rate,
            "torque": rew_torque,
            "joint_acc": rew_joint_acc,
            "action_rate": rew_action_rate,
            "orientation": rew_orientation,
            "base_height": rew_base_height,
            "feet_air_time": rew_feet_air_time,
            "alive": rew_alive,
        }

        # Termination
        self.max_body_tilt = max_body_tilt
        self.min_base_height = min_base_height

        # Spaces
        obs_dim = 3 + 3 + 12 + 12 + 12 + 3  # 45
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )

        # Internal state
        self._step_count = 0
        self._last_action = np.zeros(NUM_JOINTS)
        self._last_joint_vel = np.zeros(NUM_JOINTS)
        self._command = np.zeros(3)  # vx, vy, yaw_rate
        self._feet_air_time = np.zeros(4)

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._init_viewer()

    def _init_viewer(self):
        """Initialize MuJoCo interactive viewer."""
        try:
            import mujoco.viewer
            self._viewer_handle = None  # Will be created in render()
        except ImportError:
            pass

    def _get_joint_pos(self) -> np.ndarray:
        return self.data.qpos[self._joint_qpos_adr]

    def _get_joint_vel(self) -> np.ndarray:
        return self.data.qvel[self._joint_dof_adr]

    def _get_base_pos(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    def _get_base_quat(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def _get_base_lin_vel(self) -> np.ndarray:
        """Get base linear velocity in body frame."""
        vel_world = self.data.qvel[:3]
        quat = self._get_base_quat()
        # Rotate world vel to body frame
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        return rot.T @ vel_world

    def _get_base_ang_vel(self) -> np.ndarray:
        """Get base angular velocity in body frame."""
        ang_vel_world = self.data.qvel[3:6]
        quat = self._get_base_quat()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        return rot.T @ ang_vel_world

    def _get_projected_gravity(self) -> np.ndarray:
        """Get gravity vector projected into body frame."""
        quat = self._get_base_quat()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        return rot.T @ gravity_world

    def _get_feet_contact(self) -> np.ndarray:
        """Check which feet are in contact with the ground."""
        contacts = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for foot_idx, geom_id in enumerate(self._foot_geom_ids):
                if (c.geom1 == geom_id and c.geom2 == self._floor_geom_id) or \
                   (c.geom2 == geom_id and c.geom1 == self._floor_geom_id):
                    contacts[foot_idx] = True
        return contacts

    def _sample_command(self):
        """Sample a random velocity command."""
        self._command = np.array([
            np.random.uniform(*self.cmd_vx_range),
            np.random.uniform(*self.cmd_vy_range),
            np.random.uniform(*self.cmd_yaw_range),
        ])

    def _compute_observation(self) -> np.ndarray:
        proj_gravity = self._get_projected_gravity()
        joint_pos = self._get_joint_pos() - self.default_joint_angles
        joint_vel = self._get_joint_vel()
        base_ang_vel = self._get_base_ang_vel()

        obs = np.concatenate([
            proj_gravity,                    # 3
            self._command,                   # 3
            joint_pos,                       # 12
            joint_vel * 0.05,                # 12 (scaled)
            self._last_action,               # 12
            base_ang_vel * 0.25,             # 3 (scaled)
        ]).astype(np.float32)

        return obs

    def _apply_action(self, action: np.ndarray):
        """Apply PD control to track target joint positions."""
        target_pos = self.default_joint_angles + action * self.action_scale
        current_pos = self._get_joint_pos()
        current_vel = self._get_joint_vel()

        torques = self.kp * (target_pos - current_pos) - self.kd * current_vel
        torques = np.clip(torques, -self._torque_limits, self._torque_limits)

        self.data.ctrl[:] = torques

    def _compute_reward(self) -> tuple[float, dict]:
        """Compute multi-term reward."""
        info = {}
        total_reward = 0.0

        base_lin_vel = self._get_base_lin_vel()
        base_ang_vel = self._get_base_ang_vel()
        joint_vel = self._get_joint_vel()
        joint_acc = (joint_vel - self._last_joint_vel) / self.control_dt
        proj_gravity = self._get_projected_gravity()
        base_height = self._get_base_pos()[2]
        contacts = self._get_feet_contact()

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
        r_action_rate = np.sum((self._last_action - self.data.ctrl / self._torque_limits.max()) ** 2)
        total_reward += self.rew_scales["action_rate"] * r_action_rate
        info["r_action_rate"] = r_action_rate

        # Orientation penalty (penalize non-upright)
        r_orientation = np.sum(proj_gravity[:2] ** 2)
        total_reward += self.rew_scales["orientation"] * r_orientation
        info["r_orientation"] = r_orientation

        # Base height penalty
        target_height = 0.27  # from keyframe
        r_height = (base_height - target_height) ** 2
        total_reward += self.rew_scales["base_height"] * r_height
        info["r_height"] = r_height

        # Feet air time reward (encourage alternating gait)
        self._feet_air_time += self.control_dt
        self._feet_air_time *= ~contacts  # reset on contact
        r_air_time = np.sum(
            (self._feet_air_time - 0.5) * contacts
        )
        r_air_time = np.clip(r_air_time, 0.0, None)
        total_reward += self.rew_scales["feet_air_time"] * r_air_time
        info["r_air_time"] = r_air_time

        # Alive bonus
        total_reward += self.rew_scales["alive"]

        info["total_reward"] = total_reward
        return total_reward, info

    def _check_termination(self) -> tuple[bool, bool]:
        """Check termination and truncation conditions."""
        proj_gravity = self._get_projected_gravity()
        base_height = self._get_base_pos()[2]

        # Body tilt too large
        body_tilt = np.arccos(np.clip(-proj_gravity[2], -1.0, 1.0))
        terminated = bool(body_tilt > self.max_body_tilt)

        # Base too low
        terminated = terminated or bool(base_height < self.min_base_height)

        # Truncated by time limit
        truncated = self._step_count >= self.max_episode_steps

        return terminated, truncated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Set to home keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Add small random perturbation to joint positions
        noise = self.np_random.uniform(-0.05, 0.05, size=NUM_JOINTS)
        self.data.qpos[self._joint_qpos_adr] += noise

        # Small random base position perturbation
        self.data.qpos[0] += self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[1] += self.np_random.uniform(-0.05, 0.05)

        mujoco.mj_forward(self.model, self.data)

        # Reset internal state
        self._step_count = 0
        self._last_action = np.zeros(NUM_JOINTS)
        self._last_joint_vel = np.zeros(NUM_JOINTS)
        self._feet_air_time = np.zeros(4)

        # Sample new command
        self._sample_command()

        obs = self._compute_observation()
        info = {"command": self._command.copy()}

        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        # Apply PD control for n_substeps
        self._apply_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Compute reward
        reward, reward_info = self._compute_reward()

        # Update internal state
        self._last_joint_vel = self._get_joint_vel().copy()
        self._last_action = action.copy()

        # Check termination
        terminated, truncated = self._check_termination()

        obs = self._compute_observation()
        info = {
            **reward_info,
            "command": self._command.copy(),
            "base_pos": self._get_base_pos(),
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
