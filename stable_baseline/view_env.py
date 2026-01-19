"""
Visualize MyPickCube-v0 environment with GUI
Run random actions and watch the robot interact with the cube
"""

import gymnasium as gym
import numpy as np
import time

# Import custom environment
import my_pick_cube_env


def view_random_actions():
    """View environment with random actions"""
    print("=" * 60)
    print("MyPickCube-v0 Environment Viewer")
    print("=" * 60)
    print("\nStarting environment with random actions...")
    print("Close the viewer window to exit.\n")
    
    # Create environment with GUI rendering
    env = gym.make(
        "MyPickCube-v0",
        num_envs=1,  # single environment for viewing
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",  # GUI mode
        sim_backend="auto",
    )
    
    obs, info = env.reset(seed=42)
    print("Environment created. GUI window should appear.")
    print("Running random actions. Watch the robot!\n")
    
    episode_num = 1
    step = 0
    episode_reward = 0.0
    
    try:
        while True:
            # Random action
            action = env.action_space.sample()
            
            # Small bias towards closing gripper occasionally
            if step % 50 == 0:
                action[-1] = -0.5  # try to close gripper
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward[0]
            step += 1
            
            # Slow down for viewing
            time.sleep(0.01)
            
            if terminated.any() or truncated.any():
                print(f"Episode {episode_num} finished:")
                print(f"  - Steps: {step}")
                print(f"  - Reward: {episode_reward:.2f}")
                print(f"  - Success: {info.get('success', [False])[0]}")
                print()
                
                episode_num += 1
                step = 0
                episode_reward = 0.0
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
    finally:
        env.close()
        print("Environment closed.")


def view_zero_actions():
    """View environment with zero actions (robot stays still)"""
    print("=" * 60)
    print("MyPickCube-v0 Environment Viewer (Zero Actions)")
    print("=" * 60)
    print("\nStarting environment with zero actions...")
    print("Robot will stay in place. You can see the initial setup.")
    print("Close the viewer window to exit.\n")
    
    env = gym.make(
        "MyPickCube-v0",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
        sim_backend="auto",
    )
    
    obs, info = env.reset(seed=42)
    print("Environment created. GUI window should appear.\n")
    
    # Create zero action with correct shape (num_envs, action_dim)
    action = np.zeros((1, env.action_space.shape[-1]))  # (1, 8)
    
    try:
        for step in range(1000):
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.02)
            
            if terminated.any() or truncated.any():
                print(f"Episode ended at step {step+1}")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
    finally:
        env.close()
        print("Environment closed.")


def view_multiple_resets():
    """View environment with multiple resets to see randomization"""
    print("=" * 60)
    print("MyPickCube-v0 Environment Viewer (Multiple Resets)")
    print("=" * 60)
    print("\nShowing environment with different random initializations...")
    print("Close the viewer window to exit.\n")
    
    env = gym.make(
        "MyPickCube-v0",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
        sim_backend="auto",
    )
    
    reset_num = 1
    
    try:
        while True:
            print(f"Reset #{reset_num}")
            obs, info = env.reset()
            
            # Hold still for 3 seconds to see the setup
            # Create zero action with correct shape (num_envs, action_dim)
            action = np.zeros((1, env.action_space.shape[-1]))
            for _ in range(150):  # 3 seconds at ~50 FPS
                obs, reward, terminated, truncated, info = env.step(action)
                time.sleep(0.02)
            
            reset_num += 1
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C)")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View MyPickCube-v0 environment")
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "zero", "resets"],
        help="Viewing mode: random actions, zero actions, or multiple resets",
    )
    
    args = parser.parse_args()
    
    print("\n🎮 Controls:")
    print("  - Press Ctrl+C to stop")
    print("  - Close the viewer window to exit\n")
    
    if args.mode == "random":
        view_random_actions()
    elif args.mode == "zero":
        view_zero_actions()
    elif args.mode == "resets":
        view_multiple_resets()
