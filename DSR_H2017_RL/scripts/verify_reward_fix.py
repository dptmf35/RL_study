#!/usr/bin/env python3
"""Verify reward function directly from environment."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv


def main():
    print("="*70)
    print("REWARD VERIFICATION (Direct from environment)")
    print("="*70)

    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())
    obs, info = env.reset()

    # Get initial state
    ee_pos = env.data.site_xpos[env.ee_site_id].copy()
    cube_pos = env.data.xpos[env.cube_body_id].copy()

    print(f"\nInitial state:")
    print(f"  EE:   ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
    print(f"  Cube: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")

    # Test scenario: Take random actions and monitor reward
    print("\n" + "-"*70)
    print("Taking 10 random steps:")
    print(f"{'Step':<6} {'Dist XY':<10} {'Height Err':<12} {'Reward':<10} {'Components'}")
    print("-"*70)

    for step in range(10):
        action = np.random.uniform(-0.5, 0.5, size=7)
        obs, reward, terminated, truncated, info = env.step(action)

        reward_comps = info.get('reward_components', {})
        comps_str = " ".join([f"{k[:3]}:{v:+.2f}" for k, v in reward_comps.items()])

        print(f"{step+1:<6} {info['distance_xy']:<10.4f} {info['height_error']:<12.4f} "
              f"{reward:<10.2f} {comps_str}")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Check if bonuses are present
    has_bonuses = any('bonus' in k for k in reward_comps.keys())

    if has_bonuses:
        print("⚠️  Bonuses detected in reward components")
        print("   Check if they're conditional on height")
    else:
        print("✅ No bonuses - pure dense reward")
        print("   Agent must minimize both distance and height penalties")

    print("\nExpected behavior:")
    print("  - Both distance and height penalties should be equally weighted (-5.0)")
    print("  - No bonuses for partial alignment")
    print("  - Only success bonus (+100) when fully aligned")

    env.close()


if __name__ == "__main__":
    main()
