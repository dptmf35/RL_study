"""
Save screenshots of the environment instead of using GUI
"""

import gymnasium as gym
import numpy as np
from PIL import Image
import os

# Import custom environment
import my_pick_cube_env


def save_initial_view():
    """Save screenshot of initial environment setup"""
    print("=" * 60)
    print("Saving Environment Screenshots")
    print("=" * 60)
    
    # Create output directory
    output_dir = "screenshots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving to: {output_dir}/\n")
    
    # Create environment with rgb_array mode (offscreen rendering)
    env = gym.make(
        "MyPickCube-v0",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",  # offscreen rendering
        sim_backend="auto",
    )
    
    print("[1/3] Resetting environment...")
    obs, info = env.reset(seed=42)
    
    # Render and save initial frame
    print("[2/3] Rendering initial view...")
    frame = env.render()
    
    if frame is not None:
        # Convert to numpy array if needed
        if hasattr(frame, 'cpu'):
            frame = frame.cpu().numpy()
        if hasattr(frame, 'numpy'):
            frame = frame.numpy()
        
        # Handle batch dimension
        if frame.ndim == 4:
            frame = frame[0]
        
        # Convert to uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Save image
        img = Image.fromarray(frame)
        filepath = f"{output_dir}/initial_view.png"
        img.save(filepath)
        print(f"✓ Saved: {filepath}")
    
    # Take a few random steps and save
    print("\n[3/3] Taking random actions and saving frames...")
    action = env.action_space.sample()
    
    for step in [10, 20, 30, 50]:
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(action)
            action = env.action_space.sample()
        
        frame = env.render()
        if frame is not None:
            if hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()
            if hasattr(frame, 'numpy'):
                frame = frame.numpy()
            if frame.ndim == 4:
                frame = frame[0]
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            img = Image.fromarray(frame)
            filepath = f"{output_dir}/step_{step:03d}.png"
            img.save(filepath)
            print(f"✓ Saved: {filepath}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("✓ Screenshots saved successfully!")
    print(f"Check the '{output_dir}/' folder")
    print("=" * 60)


def save_multiple_resets():
    """Save screenshots of multiple random initializations"""
    print("=" * 60)
    print("Saving Multiple Environment Initializations")
    print("=" * 60)
    
    output_dir = "screenshots/resets"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving to: {output_dir}/\n")
    
    env = gym.make(
        "MyPickCube-v0",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        sim_backend="auto",
    )
    
    num_resets = 5
    for i in range(num_resets):
        print(f"Reset #{i+1}/{num_resets}...")
        obs, info = env.reset()
        
        frame = env.render()
        if frame is not None:
            if hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()
            if hasattr(frame, 'numpy'):
                frame = frame.numpy()
            if frame.ndim == 4:
                frame = frame[0]
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            img = Image.fromarray(frame)
            filepath = f"{output_dir}/reset_{i+1:02d}.png"
            img.save(filepath)
            print(f"✓ Saved: {filepath}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("✓ Multiple resets saved!")
    print(f"Check the '{output_dir}/' folder")
    print("You can see how cube positions vary")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Save environment screenshots")
    parser.add_argument(
        "--mode",
        type=str,
        default="initial",
        choices=["initial", "resets"],
        help="Save initial view with actions, or multiple resets",
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "initial":
            save_initial_view()
        elif args.mode == "resets":
            save_multiple_resets()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
