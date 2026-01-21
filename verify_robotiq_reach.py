#!/usr/bin/env python3
"""
Verify that UR5e + Robotiq can physically reach and grasp the cube
"""

import numpy as np
import mujoco
from pathlib import Path


def verify_reach():
    """Verify robot can reach cube positions."""
    model_path = Path(__file__).parent / "assets" / "ur5e_robotiq_pick_place.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    print("="*60)
    print("UR5e + Robotiq 2F85 Physical Reachability Verification")
    print("="*60)
    
    # Get IDs
    pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    
    # Reset to home position
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Get positions
    base_pos = data.xpos[base_body_id]
    pinch_pos = data.site_xpos[pinch_site_id]
    cube_pos = data.xpos[cube_body_id]
    
    print(f"\n📍 Positions:")
    print(f"  Robot Base:  {base_pos}")
    print(f"  Pinch Site:  {pinch_pos}")
    print(f"  Cube:        {cube_pos}")
    
    # Calculate distances
    base_to_cube = np.linalg.norm(cube_pos - base_pos)
    pinch_to_cube = np.linalg.norm(cube_pos - pinch_pos)
    
    print(f"\n📏 Distances:")
    print(f"  Base to Cube:  {base_to_cube:.3f} m")
    print(f"  Pinch to Cube: {pinch_to_cube:.3f} m")
    
    # UR5e specifications
    ur5e_max_reach = 0.85  # meters
    
    print(f"\n🤖 UR5e Specifications:")
    print(f"  Max Reach: {ur5e_max_reach} m")
    
    # Gripper specifications
    robotiq_max_opening = 0.085  # 85mm
    cube_size = 0.03  # 30mm (full width)
    
    print(f"\n✋ Robotiq 2F85 Gripper:")
    print(f"  Max Opening: {robotiq_max_opening*1000:.0f} mm")
    print(f"  Cube Width:  {cube_size*1000:.0f} mm")
    
    # Test spawn ranges
    print(f"\n🎯 Cube Spawn Ranges:")
    
    # Easy mode
    easy_range = {
        "x": (0.45, 0.55),
        "y": (-0.05, 0.05),
        "z": 0.435
    }
    
    print(f"\n  Easy Mode:")
    print(f"    X: {easy_range['x']}")
    print(f"    Y: {easy_range['y']}")
    print(f"    Z: {easy_range['z']}")
    
    # Check corners
    corners = [
        (easy_range['x'][0], easy_range['y'][0], easy_range['z']),  # closest
        (easy_range['x'][1], easy_range['y'][1], easy_range['z']),  # farthest
        (easy_range['x'][0], easy_range['y'][1], easy_range['z']),  # left-front
        (easy_range['x'][1], easy_range['y'][0], easy_range['z']),  # right-back
    ]
    
    print(f"\n  Easy Mode Corner Distances:")
    for i, corner in enumerate(corners):
        dist = np.linalg.norm(np.array(corner) - base_pos)
        reachable = "✅" if dist < ur5e_max_reach else "❌"
        print(f"    Corner {i+1}: {dist:.3f} m {reachable}")
    
    # Normal mode
    normal_range = {
        "x": (0.4, 0.6),
        "y": (-0.15, 0.15),
        "z": 0.435
    }
    
    print(f"\n  Normal Mode:")
    print(f"    X: {normal_range['x']}")
    print(f"    Y: {normal_range['y']}")
    print(f"    Z: {normal_range['z']}")
    
    corners = [
        (normal_range['x'][0], normal_range['y'][0], normal_range['z']),
        (normal_range['x'][1], normal_range['y'][1], normal_range['z']),
        (normal_range['x'][0], normal_range['y'][1], normal_range['z']),
        (normal_range['x'][1], normal_range['y'][0], normal_range['z']),
    ]
    
    print(f"\n  Normal Mode Corner Distances:")
    max_dist = 0
    for i, corner in enumerate(corners):
        dist = np.linalg.norm(np.array(corner) - base_pos)
        max_dist = max(max_dist, dist)
        reachable = "✅" if dist < ur5e_max_reach else "❌"
        print(f"    Corner {i+1}: {dist:.3f} m {reachable}")
    
    # Final verdict
    print(f"\n{'='*60}")
    print(f"📊 Verification Results:")
    print(f"{'='*60}")
    
    if base_to_cube < ur5e_max_reach:
        print(f"✅ Default cube position IS reachable")
    else:
        print(f"❌ Default cube position NOT reachable")
    
    if max_dist < ur5e_max_reach:
        print(f"✅ All spawn positions ARE reachable")
    else:
        print(f"❌ Some spawn positions NOT reachable")
    
    if cube_size < robotiq_max_opening:
        print(f"✅ Cube CAN be grasped by gripper")
    else:
        print(f"❌ Cube TOO LARGE for gripper")
    
    # Gripper clearance
    clearance = robotiq_max_opening - cube_size
    print(f"\n✋ Gripper Clearance: {clearance*1000:.1f} mm")
    
    if clearance > 0.004:  # 4mm safety margin
        print(f"✅ Sufficient clearance for grasping")
    else:
        print(f"⚠️  Tight fit - may be difficult to grasp")
    
    print(f"\n{'='*60}")
    print("Verification Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    verify_reach()
