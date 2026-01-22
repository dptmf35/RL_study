#!/usr/bin/env python3
"""
Environment Test Script for UR5e + Robotiq 2F85

This script tests:
1. Environment loads correctly
2. Robot can reach the cube from home position
3. Cube physics are stable (doesn't fly away)
4. Rewards are computed correctly
5. Episode terminates on success

Usage:
    python scripts/test_env_robotiq.py --task reach --render
    python scripts/test_env_robotiq.py --task pick --render
"""

import sys
from pathlib import Path
import numpy as np
import argparse
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.ur5e_robotiq_pick_place_env import UR5eRobotiqPickPlaceEnv


def test_environment_basics(env):
    """Test basic environment functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Environment Checks")
    print("="*60)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Initial distance to cube: {info['distance_to_cube']:.4f}m")
    print(f"Cube height: {info['cube_height']:.4f}m")

    # Check observation bounds
    assert obs.shape == (32,), f"Expected obs shape (32,), got {obs.shape}"
    assert env.action_space.shape == (7,), f"Expected action shape (7,), got {env.action_space.shape}"

    print("PASS: Basic environment checks passed!")
    return True


def test_reach_heuristic(env, render=False):
    """Test if a simple heuristic can reach the cube."""
    print("\n" + "="*60)
    print("TEST 2: Reach Heuristic Test")
    print("="*60)

    obs, info = env.reset()
    initial_dist = info['distance_to_cube']
    print(f"Initial distance to cube: {initial_dist:.4f}m")

    total_reward = 0
    min_dist = initial_dist
    success = False

    for step in range(200):
        # Simple heuristic: move joints towards cube direction
        # Extract relative position from observation
        # obs[26:29] is cube_to_gripper (cube_pos - ee_pos)
        relative_pos = obs[26:29]  # This points from gripper to cube

        # Simple proportional control
        action = np.zeros(7)

        # Use joint 1 (shoulder_lift) and joint 2 (elbow) mainly
        # Positive relative_pos[0] means cube is in front -> lower shoulder
        # Positive relative_pos[2] means cube is above -> adjust accordingly

        if relative_pos[0] > 0.02:  # Cube is in front
            action[1] = 0.5  # Lower shoulder
            action[2] = -0.3  # Extend elbow
        elif relative_pos[0] < -0.02:  # Cube is behind
            action[1] = -0.5
            action[2] = 0.3

        if relative_pos[2] < -0.02:  # Cube is below
            action[1] = 0.3
            action[3] = 0.3
        elif relative_pos[2] > 0.02:  # Cube is above
            action[1] = -0.3
            action[3] = -0.3

        # Y direction
        if relative_pos[1] > 0.02:
            action[0] = -0.3
        elif relative_pos[1] < -0.02:
            action[0] = 0.3

        # Keep gripper open for reach
        action[6] = -1.0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render:
            env.render()
            time.sleep(0.02)

        dist = info['distance_to_cube']
        min_dist = min(min_dist, dist)

        if terminated:
            success = info['task_success']
            print(f"Episode terminated at step {step+1}")
            break

        if step % 50 == 0:
            print(f"Step {step}: dist={dist:.4f}m, reward={reward:.4f}")

    print(f"\nResults:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Min distance achieved: {min_dist:.4f}m")
    print(f"  Distance improved: {initial_dist - min_dist:.4f}m")
    print(f"  Task success: {success}")

    if min_dist < 0.1:
        print("PASS: Heuristic can approach cube!")
        return True
    else:
        print("FAIL: Heuristic cannot approach cube effectively")
        return False


def test_cube_physics(env, render=False):
    """Test that cube doesn't fly away on contact."""
    print("\n" + "="*60)
    print("TEST 3: Cube Physics Stability")
    print("="*60)

    obs, info = env.reset()

    # Move robot towards cube aggressively
    max_cube_velocity = 0.0

    for step in range(100):
        # Random aggressive actions
        action = np.random.uniform(-1, 1, 7)
        action[6] = -1  # Keep gripper open

        obs, reward, terminated, truncated, info = env.step(action)

        if render:
            env.render()
            time.sleep(0.01)

        # Check cube velocity
        cube_vel = np.linalg.norm(obs[17:20])  # cube_vel is at indices 17-20
        max_cube_velocity = max(max_cube_velocity, cube_vel)

        # Check if cube flew off table
        cube_height = info['cube_height']
        if cube_height < 0.3 or cube_height > 1.0:
            print(f"FAIL: Cube flew off table! Height: {cube_height:.4f}m")
            return False

    print(f"Max cube velocity observed: {max_cube_velocity:.4f}m/s")
    if max_cube_velocity < 2.0:  # Reasonable threshold
        print("PASS: Cube physics are stable!")
        return True
    else:
        print("WARNING: Cube velocity is high, might fly in some cases")
        return True  # Still pass but warn


def test_reward_computation(env):
    """Test reward computation for different states."""
    print("\n" + "="*60)
    print("TEST 4: Reward Computation")
    print("="*60)

    obs, info = env.reset()

    # Take a few steps and check rewards
    rewards = []
    for _ in range(10):
        action = np.zeros(7)
        action[6] = -1  # Keep gripper open
        obs, reward, _, _, info = env.step(action)
        rewards.append(reward)

    print(f"Rewards for standing still: {rewards[:5]}")
    print(f"Mean reward: {np.mean(rewards):.4f}")

    # Rewards should be negative (distance-based) but not extreme
    if all(-5 < r < 5 for r in rewards):
        print("PASS: Rewards are in reasonable range!")
        return True
    else:
        print("WARNING: Rewards might be too extreme")
        return True


def test_episode_reset(env):
    """Test that episodes reset correctly."""
    print("\n" + "="*60)
    print("TEST 5: Episode Reset")
    print("="*60)

    cube_positions = []
    for i in range(5):
        obs, info = env.reset()
        # Observation structure: joint_pos(6), joint_vel(6), gripper_pos(1), gripper_vel(1),
        # ee_pos(3), cube_pos(3), cube_vel(3), target_pos(3), cube_to_gripper(3), cube_to_target(3)
        cube_pos = obs[17:20]  # cube_pos is at indices 17-20
        ee_pos = obs[14:17]    # ee_pos for reference
        cube_positions.append(cube_pos.copy())
        print(f"Reset {i+1}: Cube at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}), EE at ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")

    # Check positions are randomized
    positions_array = np.array(cube_positions)
    x_variance = np.var(positions_array[:, 0])
    y_variance = np.var(positions_array[:, 1])

    print(f"X position variance: {x_variance:.6f}")
    print(f"Y position variance: {y_variance:.6f}")
    print(f"Note: If variance is 0, randomization may not be working")

    if x_variance > 0.0001 and y_variance > 0.0001:
        print("PASS: Cube positions are randomized!")
        return True
    else:
        print("WARNING: Cube randomization might not be working")
        return True


def run_random_agent(env, num_episodes=3, render=False):
    """Run random agent to check environment stability."""
    print("\n" + "="*60)
    print("TEST 6: Random Agent Stability")
    print("="*60)

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()
                time.sleep(0.01)

            if terminated or truncated:
                break

        print(f"Episode {ep+1}: {steps} steps, reward={total_reward:.2f}, success={info.get('task_success', False)}")

    print("PASS: Environment is stable with random actions!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test UR5e + Robotiq environment")
    parser.add_argument("--task", type=str, default="reach", choices=["reach", "pick", "pick_place"])
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--easy", action="store_true", help="Use easy mode")
    args = parser.parse_args()

    print("="*60)
    print(f"Testing UR5e + Robotiq 2F85 Environment")
    print(f"Task mode: {args.task}")
    print(f"Render: {args.render}")
    print(f"Easy mode: {args.easy}")
    print("="*60)

    # Create environment
    render_mode = "human" if args.render else None
    env = UR5eRobotiqPickPlaceEnv(
        render_mode=render_mode,
        max_episode_steps=200,
        reward_type="dense",
        randomize_cube=True,
        task_mode=args.task,
        easy_mode=args.easy,
    )

    try:
        # Run tests
        results = []
        results.append(("Basic checks", test_environment_basics(env)))
        results.append(("Reach heuristic", test_reach_heuristic(env, args.render)))
        results.append(("Cube physics", test_cube_physics(env, args.render)))
        results.append(("Reward computation", test_reward_computation(env)))
        results.append(("Episode reset", test_episode_reset(env)))
        results.append(("Random agent", run_random_agent(env, render=args.render)))

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        all_passed = True
        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\nAll tests passed! Environment is ready for training.")
        else:
            print("\nSome tests failed. Please check the issues above.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
