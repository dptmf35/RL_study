# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Applications

Scripts must be run through Isaac Sim's Python interpreter, not the system Python:

```bash
cd /workspaces/IsaacSim
./python.sh /workspaces/IsaacRobotics/applications/spot_warehouse.py
```

There is no build step, linting configuration, or test suite — this is a simulation scripting project.

## Architecture

### Policy Controllers (`applications/spot_policy.py`)

Two policy classes extend `PolicyController` from `isaacsim.robot.policy.examples.controllers`:

- **`SpotFlatTerrainPolicy`** — Spot quadruped (12 DoF). Observation vector: 48-dim `[lin_vel(3), ang_vel(3), gravity_b(3), command(3), joint_pos(12), joint_vel(12), prev_action(12)]`.
- **`SpotArmFlatTerrainPolicy`** — Spot with arm (19 DoF). Observation vector: 69-dim `[lin_vel(3), ang_vel(3), gravity_b(3), command(3), joint_pos(19), joint_vel(19), prev_action(19)]`.

Both classes:
1. Load a `.pt` policy model and `env.yaml` params via `self.load_policy(policy_path, policy_params_path)` in `__init__`.
2. Run inference every `self._decimation` physics steps in `forward(dt, command)`.
3. Apply joint position targets: `default_pos + (action * action_scale)` where `action_scale = 0.2`.
4. Commands are `np.ndarray([v_x, v_y, w_z])` — linear x, linear y, angular z velocities in the robot body frame.

Velocities and gravity are transformed from world frame to body frame using the rotation matrix derived from `quat_to_rot_matrix`.

### Application Loop (`applications/spot_warehouse.py`)

`SpotRunner` wires together:
- Isaac Sim `World` (physics_dt=1/200s, render_dt=1/60s)
- The warehouse USD environment from Isaac Sim assets
- `SpotArmFlatTerrainPolicy` loaded from `policies/spot_arm/`
- Keyboard event handler that modulates the 3-element command vector

Physics callback `on_physics_step` handles first-step initialization, reset, and per-step `forward()` calls.

### File Layout

```
assets/               — USD robot models (spot.usd, spot_arm.usd, spot_arm.usda)
policies/
  spot/
    models/spot_policy.pt
    params/env.yaml
  spot_arm/
    models/spot_arm_policy.pt
    params/env.yaml, agent.yaml, agent.pkl, env.pkl
applications/
  spot_policy.py      — PolicyController subclasses
  spot_warehouse.py   — Runnable warehouse demo
```

### Isaac Sim API Conventions

- Prims are placed on the USD stage with paths like `/World/Spot`.
- `enable_extension('isaacsim.ros2.bridge')` enables optional ROS 2 integration (requires ROS 2 Humble + rmw_zenoh).
- `get_assets_root_path()` resolves the Isaac Sim built-in assets directory for environment USDs.
- `ArticulationAction(joint_positions=...)` is used to command joint targets.
