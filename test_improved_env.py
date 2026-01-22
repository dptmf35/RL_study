"""Test the improved UR5e Robotiq Pick Place environment"""
import sys
sys.path.append('/home/yeseul/Desktop/mygitrepos/RL_study')

from envs.ur5e_robotiq_pick_place_env import UR5eRobotiqPickPlaceEnv
import numpy as np

print("="*80)
print("Testing Improved UR5e Robotiq Environment")
print("="*80)

# Create environment
env = UR5eRobotiqPickPlaceEnv(
    task_mode='pick_place',
    easy_mode=True,  # Use easy mode for testing
    max_episode_steps=100
)

print(f"\nEnvironment created!")
print(f"Task mode: {env.task_mode}")
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.shape}")

# Reset and get initial state
obs, info = env.reset(seed=42)
print(f"\nInitial state:")
print(f"  Distance to cube: {info['distance_to_cube']:.4f} m")
print(f"  Distance to target: {info['distance_to_target']:.4f} m")
print(f"  Cube height: {info['cube_height']:.4f} m")
print(f"  Is grasped: {info['is_grasped']}")
print(f"  Is obj placed: {info['is_obj_placed']}")
print(f"  Is robot static: {info['is_robot_static']}")

# Run a few steps with random actions
print(f"\nRunning 20 steps with random actions...")
print("-"*80)

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step % 5 == 0:
        print(f"\nStep {step}:")
        print(f"  Reward: {reward:.4f}")
        if 'reward_components' in info:
            for key, val in info['reward_components'].items():
                print(f"    {key}: {val:.4f}")
        print(f"  Distance to cube: {info['distance_to_cube']:.4f}")
        print(f"  Is grasped: {info['is_grasped']}")
        print(f"  Is obj placed: {info['is_obj_placed']}")
        print(f"  Is robot static: {info['is_robot_static']}")
    
    if terminated or truncated:
        print(f"\n{'='*80}")
        print(f"Episode ended at step {step+1}")
        print(f"Task success: {info['task_success']}")
        print(f"Final reward: {reward:.4f}")
        break

env.close()
print(f"\n{'='*80}")
print("Test complete!")
print("="*80)
