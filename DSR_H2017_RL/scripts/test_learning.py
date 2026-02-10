#!/usr/bin/env python3
"""Quick test to verify environment learning capability."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv


def test_environment_properties():
    """Test key environment properties for learning."""
    print("="*70)
    print("ENVIRONMENT LEARNING CAPABILITY TEST")
    print("="*70)

    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())

    # Test 1: Initial distance distribution
    print("\n1. Initial Distance Distribution (10 resets)")
    print("-" * 70)
    distances = []
    for i in range(10):
        obs, info = env.reset()
        distances.append(info['distance_xy'])
        print(f"   Episode {i+1}: {info['distance_xy']:.3f}m "
              f"(height_error: {info['height_error']:.3f}m)")

    print(f"\n   Mean initial distance: {np.mean(distances):.3f}m ± {np.std(distances):.3f}m")
    print(f"   Min: {np.min(distances):.3f}m, Max: {np.max(distances):.3f}m")

    # Test 2: Action effect analysis
    print("\n2. Action Effect Analysis")
    print("-" * 70)
    env.reset()

    # Test small positive action on each joint
    initial_obs = env._get_obs()
    initial_info = env._get_info()

    print(f"   Initial distance: {initial_info['distance_xy']:.3f}m")

    # Take a few steps with small random actions
    rewards = []
    for step in range(5):
        action = np.random.uniform(-0.3, 0.3, size=7)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        print(f"   Step {step+1}: reward={reward:7.2f}, distance={info['distance_xy']:.3f}m")

    print(f"\n   Mean step reward: {np.mean(rewards):.2f}")

    # Test 3: Reward range analysis
    print("\n3. Reward Range Analysis (random policy, 20 episodes)")
    print("-" * 70)
    episode_rewards = []
    episode_distances = []
    best_reward = -np.inf
    best_episode = -1

    for ep in range(20):
        obs, info = env.reset()
        ep_reward = 0
        min_distance = info['distance_xy']

        for step in range(300):
            action = np.random.uniform(-1, 1, size=7)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            min_distance = min(min_distance, info['distance_xy'])

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        episode_distances.append(min_distance)

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_episode = ep

    print(f"   Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   Min reward: {np.min(episode_rewards):.2f}")
    print(f"   Max reward: {np.max(episode_rewards):.2f}")
    print(f"   Best episode: #{best_episode+1} with reward {best_reward:.2f}")
    print(f"\n   Mean min distance: {np.mean(episode_distances):.3f}m")
    print(f"   Best min distance: {np.min(episode_distances):.3f}m")

    # Test 4: Success possibility check
    print("\n4. Success Possibility Check")
    print("-" * 70)

    # Check if any episode got close
    success_threshold = 0.04  # 4cm
    close_episodes = sum(1 for d in episode_distances if d < success_threshold)

    print(f"   Episodes reaching < 4cm: {close_episodes}/20")

    if close_episodes > 0:
        print("   ✅ Task appears learnable - random policy can reach goal!")
    else:
        closest = np.min(episode_distances)
        print(f"   ⚠️  Closest: {closest:.3f}m - may need tuning")

    # Test 5: Home position quality
    print("\n5. Home Position Quality")
    print("-" * 70)
    obs, info = env.reset()

    ee_pos = env.data.site_xpos[env.ee_site_id]
    cube_pos = env.data.xpos[env.cube_body_id]

    # Check gripper orientation
    ee_xmat = env.data.site_xmat[env.ee_site_id].reshape(3, 3)
    z_axis = ee_xmat[:, 2]
    downward = np.dot(z_axis, [0, 0, -1])

    print(f"   End-effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
    print(f"   Cube center: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
    print(f"   Initial distance: {info['distance_xy']:.3f}m")
    print(f"   Downward alignment: {downward:.3f} (1.0 = perfect)")

    if downward > 0.7:
        print("   ✅ Gripper orientation: Good")
    else:
        print("   ⚠️  Gripper orientation: Needs improvement")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    learnable = True
    issues = []

    if np.mean(distances) > 0.5:
        issues.append("Initial distance too large (>0.5m)")
        learnable = False

    if downward < 0.6:
        issues.append("Poor gripper orientation (<0.6)")

    if close_episodes == 0 and np.min(episode_distances) > 0.10:
        issues.append("Random policy never gets close (<10cm)")
        learnable = False

    if np.std(episode_rewards) < 10:
        issues.append("Low reward variance - weak learning signal")

    if learnable and len(issues) == 0:
        print("✅ Environment appears HIGHLY LEARNABLE")
        print("   - Good initial position")
        print("   - Strong reward signal")
        print("   - Random policy can reach goal")
    elif learnable:
        print("✅ Environment appears LEARNABLE with notes:")
        for issue in issues:
            print(f"   ⚠️  {issue}")
    else:
        print("❌ Environment may have LEARNING DIFFICULTIES:")
        for issue in issues:
            print(f"   ❌ {issue}")

    print("="*70)

    env.close()


if __name__ == "__main__":
    test_environment_properties()
