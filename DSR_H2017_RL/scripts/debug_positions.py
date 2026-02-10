#!/usr/bin/env python3
"""Debug script to check initial positions of robot, table, and cube."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from DSR_H2017_RL.envs import AlignmentConfig, DSRH2017AlignEnv
import numpy as np

def main():
    env = DSRH2017AlignEnv(render_mode=None, config=AlignmentConfig())

    print("="*60)
    print("INITIAL POSITION DEBUG")
    print("="*60)

    for episode in range(3):
        obs, info = env.reset()

        # Get positions
        ee_pos = env.data.site_xpos[env.ee_site_id]
        cube_pos = env.data.xpos[env.cube_body_id]
        table_pos = env.data.xpos[env.model.body("align_table").id]

        print(f"\nEpisode {episode + 1}:")
        print(f"  Robot base: (0, 0, 0)")
        print(f"  Table body: ({table_pos[0]:.3f}, {table_pos[1]:.3f}, {table_pos[2]:.3f})")
        print(f"  End effector: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
        print(f"  Cube: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
        print(f"  Distance XY: {np.linalg.norm(ee_pos[:2] - cube_pos[:2]):.3f}m")
        print(f"  Distance 3D: {np.linalg.norm(ee_pos - cube_pos):.3f}m")

        # Check joint positions
        joint_pos = np.array([env.data.qpos[env.model.jnt_qposadr[jid]]
                              for jid in env.arm_joint_ids])
        print(f"  Joint angles: {np.array2string(joint_pos, precision=3, suppress_small=True)}")

    env.close()
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
