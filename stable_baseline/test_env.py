"""
Quick test script to verify custom environment works
"""

import gymnasium as gym
import numpy as np

# Import custom environment
import my_pick_cube_env


def test_environment():
    """Test that the environment can be created and run"""
    
    print("=" * 60)
    print("Testing MyPickCube-v0 Environment")
    print("=" * 60)
    
    # Create environment
    print("\n[1/4] Creating environment...")
    env = gym.make(
        "MyPickCube-v0",
        num_envs=2,  # small number for testing
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,  # no rendering for quick test
    )
    print(f"   ✓ Environment created")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    
    # Reset
    print("\n[2/4] Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Reset successful")
    print(f"   - Observation keys: {obs.keys() if isinstance(obs, dict) else 'single array'}")
    if isinstance(obs, dict):
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"     • {key}: shape {value.shape}")
    
    # Run random actions
    print("\n[3/4] Running random actions...")
    total_reward = 0.0
    success_count = 0
    episode_count = 0
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert reward to numpy if it's a tensor
        if hasattr(reward, 'cpu'):
            reward = reward.cpu().numpy()
        total_reward += float(reward.mean())
        
        if terminated.any() or truncated.any():
            if hasattr(terminated, 'sum'):
                success_count += int(terminated.sum())
            episode_count += 1
            print(f"   Episode ended at step {step+1}")
            obs, info = env.reset()
    
    print(f"   ✓ Completed 50 steps")
    print(f"   - Total reward: {total_reward:.2f}")
    print(f"   - Average reward per step: {total_reward/50:.2f}")
    print(f"   - Episodes: {episode_count}")
    print(f"   - Successes: {success_count}")
    
    # Cleanup
    print("\n[4/4] Closing environment...")
    env.close()
    print(f"   ✓ Environment closed")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nEnvironment is ready for training.")
    print("Run: python train_ppo_pickcube.py")


if __name__ == "__main__":
    try:
        test_environment()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error above and fix the environment.")
