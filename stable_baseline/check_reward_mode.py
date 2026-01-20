"""
Check which reward mode was used during training
"""

import gymnasium as gym
import my_pick_cube_env

# Test with sparse (default)
env_sparse = gym.make("MyPickCube-v0", num_envs=1, obs_mode="state")
obs, _ = env_sparse.reset()
action = env_sparse.action_space.sample()
obs, reward_sparse, _, _, info = env_sparse.step(action)
env_sparse.close()

# Test with dense
env_dense = gym.make("MyPickCube-v0", num_envs=1, obs_mode="state", reward_mode="dense")
obs, _ = env_dense.reset()
action = env_dense.action_space.sample()
obs, reward_dense, _, _, info = env_dense.step(action)
env_dense.close()

print("=" * 60)
print("Reward Mode Check")
print("=" * 60)
print(f"Sparse reward (default): {reward_sparse[0]:.4f}")
print(f"Dense reward:            {reward_dense[0]:.4f}")
print()

if abs(reward_sparse[0]) < 0.01:
    print("⚠️  Sparse reward is near 0 (expected)")
else:
    print("✓ Sparse reward has value")

if abs(reward_dense[0]) > 0.5:
    print("✓ Dense reward is working!")
else:
    print("⚠️  Dense reward seems low")

print("=" * 60)
print("\n💡 Training reward was ~5.81")
print("If that matches sparse reward pattern,")
print("you need to re-train with reward_mode='dense'")
