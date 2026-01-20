"""
Custom ManiSkill environment: PANDA robot picking up a cube
Based on ManiSkill tabletop template
"""

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

@register_env("MyPickCube-v0", max_episode_steps=150)  # 300 -> 400, even more time
class MyPickCubeEnv(BaseEnv):
    """
    Simple pick cube task with PANDA robot.
    
    Goal: Pick up the cube and lift it to a target height.
    Observation: state-based (robot qpos/qvel, cube pos/vel, etc.)
    Action: pd_joint_delta_pos (delta joint positions)
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: "PandaAgent"  # type hint
    
    cube_half_size = 0.025  # 0.02 -> 0.025 (25% bigger, easier to grasp!)
    goal_height = 0.25  # target lift height (0.3 -> 0.2, easier)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # Basic camera for visualization (not used in state obs)
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # Load table and floor
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Create cube actor with initial pose
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([1, 0, 0, 1]),  # red cube
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),  # on table surface
        )

        # Create goal region (visual only, not physical) with initial pose
        self.goal_region = actors.build_cube(
            self.scene,
            half_size=0.03,
            color=np.array([0, 1, 0, 0.3]),  # transparent green
            name="goal_region",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.goal_height]),  # at goal height
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize cube spawn position on table
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1  # random x,y on table
            xyz[:, 2] = self.cube_half_size  # rest on table
            q = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cube.set_pose(Pose.create_from_pq(p=xyz, q=q))

            # Set goal region position (above spawn)
            goal_xyz = xyz.clone()
            goal_xyz[:, 2] = self.goal_height
            self.goal_region.set_pose(Pose.create_from_pq(p=goal_xyz, q=[1, 0, 0, 0]))
            
            # Note: table_scene.initialize() already resets robot to default state
            # Gripper initialization handled by default PANDA config
            # Reward function uses STATE-based (not action) so it will guide correct behavior

    def evaluate(self) -> dict:
        """Check if task is successful - ManiSkill style"""
        cube_height = self.cube.pose.p[:, 2]
        
        # Success conditions (like ManiSkill):
        # 1. Cube is lifted to goal height (with tolerance)
        is_lifted = cube_height >= (self.goal_height - 0.05)
        
        # 2. Cube is grasped
        is_grasped = self.agent.is_grasping(self.cube)
        
        # 3. Robot is static (qvel < 0.2)
        is_robot_static = self.agent.is_static(0.2)
        
        return {
            "success": is_lifted & is_robot_static,
            "is_lifted": is_lifted,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
        }

    def _get_obs_extra(self, info: dict):
        """Extra observations for state mode"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        
        if "state" in self.obs_mode:
            # Add cube pose and velocity
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                cube_vel=self.cube.linear_velocity,
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                cube_to_goal_pos=self.goal_region.pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        """
        Simple 4-stage dense reward (ManiSkill PickCube-v1 style)
        
        Stage 1: Reach cube (0~1)
        Stage 2: Grasp cube (0~1)
        Stage 3: Lift cube (0~1, only if grasped)
        Stage 4: Static (0~1, only if lifted)
        Success bonus: +5
        
        Total max: 5
        """
        
        # Stage 1: Reaching reward (0~1)
        tcp_to_cube_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_cube_dist)
        reward = reaching_reward
        
        # Stage 2: Grasping reward (0~1)
        is_grasped = info["is_grasped"]
        reward = reward + is_grasped.float()
        
        # Stage 3: Lifting reward (0~1, only if grasped!)
        cube_height = self.cube.pose.p[:, 2]
        cube_to_goal_dist = torch.abs(cube_height - self.goal_height)
        lift_reward = 1 - torch.tanh(5.0 * cube_to_goal_dist)
        reward = reward + lift_reward * is_grasped.float()
        
        # Stage 4: Static reward (0~1, only if lifted!)
        qvel = self.agent.robot.get_qvel()[..., :-2]  # exclude gripper joints
        static_reward = 1 - torch.tanh(5.0 * torch.linalg.norm(qvel, axis=1))
        reward = reward + static_reward * info["is_lifted"].float()
        
        # Success bonus (+5)
        reward[info["success"]] = 5.0
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        # max: 1 + 1 + 1 + 1 + success(5) = 5
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
