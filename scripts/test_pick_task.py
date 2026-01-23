#!/usr/bin/env python3
"""
Test script for pick task with updated gripper logic.

This tests that:
1. Gripper closes when EE is horizontally aligned with cube
2. Gripper stays closed once closed (latch behavior)
3. Cube gets lifted when moving to goal (above cube)
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.ur5e_robotiq_goal_env import UR5eRobotiqGoalEnv


def test_pick_task():
    """Test the pick task with visualization."""
    print("=" * 60)
    print("PICK TASK TEST")
    print("=" * 60)

    env = UR5eRobotiqGoalEnv(
        render_mode="human",
        task_mode="pick",
        easy_mode=True,
        reward_type="sparse",
    )

    obs, info = env.reset()

    print(f"\nInitial State:")
    print(f"  EE position: {obs['achieved_goal']}")
    print(f"  Goal (above cube): {obs['desired_goal']}")
    print(f"  Distance to goal: {info['distance']:.4f}")

    cube_initial_z = env._get_cube_position()[2]
    print(f"  Cube Z: {cube_initial_z:.4f}")

    # Run for 100 steps with simple heuristic action
    # Move towards goal (which is above cube)
    gripper_closed_step = None
    max_cube_height = cube_initial_z

    for step in range(150):
        # Simple heuristic: move EE towards goal
        ee_pos = env._get_ee_position()
        goal_pos = env.goal

        # Direction to goal (normalized)
        direction = goal_pos - ee_pos
        direction_norm = np.linalg.norm(direction)

        if direction_norm > 0.01:
            direction = direction / direction_norm
        else:
            direction = np.zeros(3)

        # Convert to joint action (simple approximation)
        # Joint 1 (shoulder_lift) affects Z
        # Joint 0 (shoulder_pan) affects X/Y rotation
        action = np.zeros(7)

        # Move towards goal in Cartesian space (approximate with joints)
        if direction[0] > 0.1:  # Need to move forward (X+)
            action[1] = 0.5  # shoulder_lift
            action[2] = -0.3  # elbow
        elif direction[0] < -0.1:  # Need to move back (X-)
            action[1] = -0.5
            action[2] = 0.3

        if direction[2] > 0.05:  # Need to move up (Z+)
            action[1] = -0.3
            action[3] = 0.3
        elif direction[2] < -0.05:  # Need to move down (Z-)
            action[1] = 0.3
            action[3] = -0.3

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Track gripper close event (phase transition)
        if env._gripper_closed and gripper_closed_step is None:
            gripper_closed_step = step
            print(f"\n  [Step {step}] Phase 0 -> 1: Gripper CLOSED!")
            print(f"    EE position: {obs['achieved_goal']}")
            print(f"    Cube position: {env._get_cube_position()}")
            print(f"    New goal (lift): {env.goal}")

        # Track max cube height
        cube_z = env._get_cube_position()[2]
        if cube_z > max_cube_height:
            max_cube_height = cube_z

        # Print progress
        if step % 30 == 0:
            cube_pos = env._get_cube_position()
            hold = env._lift_hold_steps
            print(f"  Step {step}: phase={env._pick_phase}, dist={info['distance']:.4f}, "
                  f"cube_z={cube_pos[2]:.4f}, hold={hold}/10, "
                  f"success={info['is_success']}")

        if terminated:
            print(f"\n  SUCCESS at step {step}!")
            break

        if truncated:
            print(f"\n  Episode truncated at step {step}")
            break

    # Final report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    final_cube_z = env._get_cube_position()[2]
    final_hold = env._lift_hold_steps
    print(f"  Gripper closed at step: {gripper_closed_step}")
    print(f"  Initial cube Z: {cube_initial_z:.4f}")
    print(f"  Final cube Z: {final_cube_z:.4f}")
    print(f"  Max cube Z: {max_cube_height:.4f}")
    print(f"  Cube lifted: {final_cube_z - cube_initial_z:.4f}m")
    print(f"  Lift hold steps: {final_hold}/10 required")
    print(f"  Success: {info['is_success']}")

    if info['is_success']:
        print("\n  PICK TASK WORKING CORRECTLY!")
    elif final_hold >= 10:
        print("\n  Held long enough but distance check failed")
    elif max_cube_height > 0.50:
        print("\n  Cube lifted to 0.50+ but not held long enough")
    elif max_cube_height > cube_initial_z + 0.02:
        print("\n  Cube was lifted but not high enough (need > 0.50m)")
    elif gripper_closed_step is not None:
        print("\n  Gripper closed but cube not lifted - check grasp physics")
    else:
        print("\n  Gripper never closed - check reach to cube")

    print("=" * 60)

    # Keep viewer open
    print("\nViewer open - press ESC to close...")
    import time
    while True:
        try:
            env.render()
            time.sleep(0.1)
        except:
            break

    env.close()


if __name__ == "__main__":
    test_pick_task()
