#!/usr/bin/env python3
"""
Quick test script for UR5e + Robotiq 2F85 environment
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

def main():
    # Load model
    model_path = Path(__file__).parent / "assets" / "ur5e_robotiq_pick_place.xml"
    print(f"Loading model from: {model_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        print("✓ Model loaded successfully!")
        
        # Print model info
        print(f"\nModel info:")
        print(f"  - DOF: {model.nv}")
        print(f"  - Actuators: {model.nu}")
        print(f"  - Bodies: {model.nbody}")
        print(f"  - Joints: {model.njnt}")
        print(f"  - Geoms: {model.ngeom}")
        
        # Print joint names
        print(f"\nJoint names:")
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                print(f"  {i}: {joint_name}")
        
        # Print actuator names
        print(f"\nActuator names:")
        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  {i}: {actuator_name}")
        
        # Set to home position
        mujoco.mj_resetDataKeyframe(model, data, 0)  # Use keyframe 0 (home)
        
        # Launch viewer
        print("\nLaunching viewer... (Press ESC to exit)")
        print("Controls:")
        print("  - Drag with left mouse: rotate view")
        print("  - Drag with right mouse: translate view")
        print("  - Scroll: zoom")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Simulation loop
            step = 0
            gripper_opening = True
            gripper_state = 0.0
            
            while viewer.is_running():
                step += 1
                
                # Simple test: slowly open/close gripper
                if step % 500 == 0:
                    gripper_opening = not gripper_opening
                
                if gripper_opening:
                    gripper_state = min(gripper_state + 1.0, 255.0)
                else:
                    gripper_state = max(gripper_state - 1.0, 0.0)
                
                # Set gripper control
                data.ctrl[6] = gripper_state  # Gripper actuator
                
                # Step simulation
                mujoco.mj_step(model, data)
                viewer.sync()
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
