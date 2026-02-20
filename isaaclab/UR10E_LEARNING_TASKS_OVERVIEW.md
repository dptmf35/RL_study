# UR10e Learning Tasks in Isaac Lab

## Scope
This summary covers UR10e tasks that are registered with an RL training config (not Play-only / ROS-only inference).

Learning task IDs:
- `Isaac-Deploy-Reach-UR10e-v0`
- `Isaac-Deploy-GearAssembly-UR10e-2F140-v0`
- `Isaac-Deploy-GearAssembly-UR10e-2F85-v0`

---

## 1) Isaac-Deploy-Reach-UR10e-v0

### Observation space (policy)
- `joint_pos` (UR10e joints)
- `joint_vel` (UR10e joints)
- `pose_command` (target EE pose command from `UniformPoseCommand`, shape `(x,y,z,qw,qx,qy,qz)`)
- Effective dimensionality is typically `6 + 6 + 7 = 19` for UR10e.

### Base algorithm
- `RSL-RL` on-policy `PPO` (non-recurrent actor-critic).
- No SAC config is registered for this task.

### Reward design
- `end_effector_keypoint_tracking` (negative distance term): `-1.5`
- `end_effector_keypoint_tracking_exp` (positive exponential shaping): `+1.5`
- `action_rate_l2`: `-0.005`
- `action_l2`: `-0.005`

### Network design
- Actor MLP: `[256, 128, 64]`, ELU
- Critic MLP: `[256, 128, 64]`, ELU
- PPO: `num_steps_per_env=512`, `max_iterations=1500`, `lr=5e-4`, `gamma=0.99`, `lam=0.95`

### Reference files
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/config/ur_10e/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/config/ur_10e/joint_pos_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/reach_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/config/ur_10e/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/mdp/rewards.py`

---

## 2) Isaac-Deploy-GearAssembly-UR10e-2F140-v0 / 3) Isaac-Deploy-GearAssembly-UR10e-2F85-v0

These two tasks share the same learning structure and differ mainly in gripper model/parameters.

### Observation space
- Policy group:
  - `joint_pos` (UR10e 6 arm joints)
  - `joint_vel` (UR10e 6 arm joints)
  - `gear_shaft_pos` (3)
  - `gear_shaft_quat` (4)
  - Approx. policy dim: `6 + 6 + 3 + 4 = 19`
- Critic group (asymmetric critic):
  - Policy terms plus `gear_pos` (3), `gear_quat` (4)
  - Approx. critic dim: `26`

### Base algorithm
- `RSL-RL` on-policy `Recurrent PPO` (LSTM actor-critic).
- No SAC config is registered for these tasks.

### Reward design
- `end_effector_gear_keypoint_tracking`: `-1.5`
- `end_effector_gear_keypoint_tracking_exp`: `+1.5`
- `action_rate_l2`: `-5e-6`
- Additional reset/termination logic for dropped gear and orientation threshold is used to stabilize training.

### Network design
- Recurrent PPO actor-critic (`RslRlPpoActorCriticRecurrentCfg`)
- Actor MLP: `[256, 128, 64]`, ELU
- Critic MLP: `[256, 128, 64]`, ELU
- RNN: `LSTM`, hidden size `256`, layers `2`
- PPO: `num_steps_per_env=512`, `max_iterations=1500`, `lr=5e-4`, `gamma=0.99`, `lam=0.95`, `num_mini_batches=16`

### Reference files
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/config/ur_10e/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/gear_assembly_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/config/ur_10e/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/mdp/rewards.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/mdp/terminations.py`

---

## Notes
- `Play` and `ROS-Inference` task IDs are for evaluation/deployment paths and are excluded from learning-task list above.
- All UR10e learning tasks currently use PPO-family configs in this repository snapshot.
