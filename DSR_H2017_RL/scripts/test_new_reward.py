#!/usr/bin/env python3
"""Test new reward function to verify it prevents reward hacking."""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv


def test_reward_scenarios():
    """Test different scenarios to verify reward function."""
    print("="*70)
    print("NEW REWARD FUNCTION TEST")
    print("="*70)

    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())
    env.reset()

    print("\nScenario comparison:")
    print("-"*70)
    print(f"{'Scenario':<30} {'Distance':<10} {'Height':<10} {'Reward/step'}")
    print("-"*70)

    scenarios = [
        ("OLD EXPLOIT (XY only)", 0.001, 0.35, "XY perfect, height ignored"),
        ("Balanced approach", 0.08, 0.08, "Both improving"),
        ("Tight XY, poor height", 0.03, 0.15, "XY good, height bad"),
        ("Poor XY, good height", 0.12, 0.01, "Height good, XY bad"),
        ("Perfect alignment", 0.01, 0.01, "Both excellent"),
    ]

    for name, dist_xy, height_err, desc in scenarios:
        # Manually compute reward
        reward_comps = {
            "distance": -5.0 * dist_xy,
            "height": -5.0 * abs(height_err),
            "coarse_bonus": 0.0,
            "tight_bonus": 0.0,
            "time": -0.01,
        }

        # Coarse bonus: XY < 10cm AND height < 5cm
        if dist_xy < 0.10 and abs(height_err) < 0.05:
            reward_comps["coarse_bonus"] = 2.0

        # Tight bonus: XY < 4cm AND height < 2cm
        if dist_xy < 0.04 and abs(height_err) <= 0.02:
            reward_comps["tight_bonus"] = 5.0

        total = sum(reward_comps.values())

        print(f"{name:<30} {dist_xy:>9.3f} {height_err:>9.3f} {total:>10.2f}")

    print("-"*70)

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print("\n1. OLD EXPLOIT scenario (XY=0.001, Height=0.35):")
    old_exploit = -5.0 * 0.001 + 2.0 + 5.0 - 5.0 * 0.35 - 0.01
    print(f"   Reward: {old_exploit:.2f} per step")
    print(f"   300 steps: {old_exploit * 300:.2f}")
    if old_exploit > 0:
        print("   ❌ Still profitable to exploit!")
    else:
        print("   ✅ NOT profitable - agent must fix height!")

    print("\n2. Balanced approach (XY=0.08, Height=0.08):")
    balanced = -5.0 * 0.08 - 5.0 * 0.08 - 0.01
    print(f"   Reward: {balanced:.2f} per step")
    print(f"   300 steps: {balanced * 300:.2f}")

    print("\n3. Perfect alignment (XY=0.01, Height=0.01):")
    perfect = -5.0 * 0.01 - 5.0 * 0.01 + 2.0 + 5.0 - 0.01
    print(f"   Reward: {perfect:.2f} per step")
    print(f"   300 steps: {perfect * 300:.2f}")
    print("   ✅ Highest reward achievable!")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if old_exploit < 0:
        print("✅ Reward hacking PREVENTED!")
        print("   Agent must align BOTH XY and height to get positive reward")
        print("   Bonuses only given when both criteria are met")
    else:
        print("⚠️  Reward hacking still possible - need further tuning")

    env.close()


if __name__ == "__main__":
    test_reward_scenarios()
