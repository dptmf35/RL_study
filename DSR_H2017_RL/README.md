# DSR H2017 Cube Alignment RL Project

This project provides a reinforcement learning environment for the Doosan **H2017** robot arm with **Robotiq 2F85** gripper in MuJoCo. The task is to **align the gripper directly above a randomly spawned cube** on a table, achieving a grasp-ready pose.

## 🎯 Task Description

- **Robot**: Doosan H2017 (6-DOF) with Robotiq 2F85 gripper
- **Objective**: Position end-effector 10cm above cube center with gripper open
- **Cube Spawning**: Random position on table (X: 0.6-0.8m, Y: ±0.2m)
- **Success Criteria**: XY distance < 4cm, height error < 2cm, gripper open
- **Episode Length**: 300 steps max

## ✅ Recent Fixes (2026-02-10)

**Issue 1**: Robot faced away from table, table legs positioned incorrectly above table top.

**Fixes Applied**:
1. **Table geometry corrected**: Legs now properly positioned below table surface (cylinder height 0.36m at z=0.36)
2. **Robot home pose optimized**: Joints configured to face table and reach cube spawn area
3. **Camera views added**: Front and side cameras for better visualization

**Issue 2**: Gripper not pointing vertically downward (perpendicular to table).

**Optimization Process** (Multi-stage):
1. Initial grid search: shoulder/elbow/wrist1 optimization → downward 0.405
2. Balanced optimization: position + orientation → downward 0.613
3. **Maximum vertical optimization** (final):
   - Maximize downward alignment with Z height constraint (0.80-0.95m)
   - Extensive search over 5 joints
   - **Final configuration**: `[0.0, -0.10, 1.80, -2.50, -1.40, 0.0]`

**Performance After All Optimizations**:
- **End-effector position**: (0.662, 0.173, 0.867) - **near-perfect accuracy**
- **Downward alignment**: **0.805** (98% improvement from 0.405, 31% from 0.613)
- **Position accuracy**: X error 0.012m, Z error 0.013m
- **Initial distance**: 0.05-0.26m (**excellent starting point!**)
- **Random policy**: Reward ~-239
  - Tighter initial distance than previous version
  - More consistent gripper orientation
- **Old trained model**: Reward ~-244 (from wrong environment)
- **Previous (broken)**: Distance 1.3-1.5m, Reward ~-1330

**Note**: H2017 kinematics create inherent arm tilt (Z variation ~0.84m) due to link geometry. The gripper vertical orientation (0.805) is prioritized over perfect arm horizontality.

## Repository Layout

```
DSR_H2017_RL/
├── assets/
│   ├── h2017/                      # Doosan robot meshes (from dsr_mujoco)
│   ├── 2f85/                       # Robotiq gripper meshes + license
│   └── h2017_2f85_merged.xml       # Combined robot + gripper + table scene
├── configs/
│   └── align_default.yaml          # Baseline PPO hyperparameters
├── envs/
│   ├── __init__.py
│   └── dsr_h2017_align_env.py      # Gymnasium environment (alignment task)
├── scripts/
│   ├── train_ppo_align.py          # PPO training entry point
│   └── evaluate_align.py           # CLI for evaluation / random rollouts
├── utils/
│   └── merge_gripper.py            # Utility used to merge MJCF assets
├── logs/                           # TensorBoard logs (created at runtime)
├── models/                         # Saved checkpoints (created at runtime)
└── videos/                         # Optional evaluation renders
```

Asset directories originate from `/home/yeseul/Desktop/doosan_ws/src/doosan-robot2/dsr_mujoco` and retain their original licensing (see `assets/2f85/LICENSE`).

## Environment Highlights

- **Observation (28 dims)**: joint positions (6), joint velocities (6), gripper driver state (1), end-effector position (3), cube position (3), cube velocity (3), relative displacement (3), normalized XY direction to cube (2), XY distance (1).
- **Action (7 dims)**: joint angle deltas (6, scaled by 0.2) + gripper command (1, open/close). The agent controls position actuators directly.
- **Reward shaping** (개선된 Dense Reward):
  - **Distance penalty**: `-5.0 × distance_xy` — 수평 거리 패널티
  - **Height penalty**: `-5.0 × |height_error|` — 높이 오차 패널티 (target: 큐브 위 10cm)
  - **Success reward**: +100.0 (최초 성공 시) / +10.0 (성공 유지 시)
  - **Time penalty**: -0.01/step — 빠른 완료 유도
  - ~~Coarse/Tight align bonuses~~ → **제거됨** (reward hacking 방지)
  - **Success 조건**: XY < 4cm, 높이 오차 ±2cm 이내, 그리퍼 open
- **Termination**: success flag triggers episode termination; otherwise episodes truncate at 300 steps.

### Reward Hacking 방지 설계

이전 버전에서 coarse align (+2.0)과 tight align (+5.0) 보너스가 있었으나, 에이전트가 **XY 정렬 보너스만 취하고 높이를 무시**하는 reward hacking이 발생했습니다.

**해결 방법**:
1. 중간 보너스(coarse/tight) 완전 제거
2. Height penalty를 -2.0에서 **-5.0으로 증가** (distance penalty와 동일 가중치)
3. 순수 dense penalty로 전환 — 성공 조건 달성 시에만 보상 부여

이를 통해 에이전트는 XY 거리와 높이를 **동시에** 최소화해야만 보상을 극대화할 수 있습니다.

## Dependencies

```bash
pip install mujoco==3.1.6 gymnasium==0.28.1 stable-baselines3==2.0.0 tensorboard
```

> **Note:** The training/evaluation scripts stub out `torch.utils.tensorboard` to avoid the heavy AWS SDK dependency from TensorBoard. Logging is disabled by default, so installing TensorBoard is optional unless you want full logging support.

## Training

Run a short PPO session (single environment) using the provided script:

```bash
python3 scripts/train_ppo_align.py \
  --total-timesteps 200000 \
  --n-envs 4 \
  --log-dir logs \
  --model-dir models
```

### PPO Hyperparameters

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `total_timesteps` | 200,000 | 총 학습 스텝 |
| `n_envs` | 4 | 병렬 환경 수 |
| `learning_rate` | 3e-4 | 학습률 |
| `n_steps` | 1024 | 업데이트당 스텝 |
| `batch_size` | 256 | 미니배치 크기 |
| `n_epochs` | 10 | 업데이트당 에폭 |
| `gamma` | 0.99 | 할인 계수 |
| `gae_lambda` | 0.95 | GAE 람다 |
| `clip_range` | 0.2 | PPO 클립 범위 |
| `ent_coef` | 0.01 | 엔트로피 계수 |
| `vf_coef` | 0.5 | Value function 계수 |
| `max_grad_norm` | 0.5 | Gradient 클리핑 |

**Network Architecture**: `[256, 256]` (Policy/Value 모두) + ReLU, `VecNormalize` 적용.

**⚠️ Note on Existing Models**: Models in `models/ppo_dsr_align_*` were trained with incorrect environment setup (robot facing wrong direction, poor gripper orientation). They should be retrained with the corrected environment.

**Expected Training Progress** (with optimized environment):
- **Random policy baseline**: ~-239 reward, 0.05-0.26m initial distance
- **Target performance**: <0.04m distance, >50% success rate
- With proper training (200k+ timesteps), should significantly outperform random policy

All checkpoints are written under `models/<run_name>/` with `best/` (highest eval reward) and `final_model.zip`, plus VecNormalize statistics.

## Evaluation & Visualisation

```bash
# Deterministic policy rollout (no rendering)
python3 scripts/evaluate_align.py \
  --model-path models/<run_name>/final_model.zip \
  --n-episodes 5 \
  --no-render

# Random policy demo with viewer
python3 scripts/evaluate_align.py --random --render --n-episodes 2
```

The evaluator automatically loads `vec_normalize.pkl` if found alongside the model.

## 🎮 Manual Teleoperation

**직접 로봇을 조작하여 최적의 home position을 찾을 수 있습니다!**

```bash
python3 scripts/teleop_dsr.py
```

키보드로 각 joint를 실시간 제어하고, 원하는 자세를 찾으면 `P` 키로 코드를 출력하여 복사할 수 있습니다.

**주요 기능**:
- 실시간 joint 제어 (키보드 1/Q, 2/W, 3/E, 4/R, 5/T, 6/Y)
- Gripper 제어 (G/H)
- 현재 상태 정보 표시 (I 키)
- 코드 출력 (P 키) - 복사하여 바로 사용
- Home position 저장/리셋 (S/SPACE)

자세한 사용법은 `TELEOPERATION.md` 참고

## Debugging & Verification

Check robot/cube positions at start:
```bash
python3 scripts/debug_positions.py
python3 scripts/debug_orientation.py  # Check gripper orientation
```

Expected output (`debug_positions.py`):
- Robot base: (0, 0, 0)
- Table: (0.7, 0, 0)
- **End effector: (0.662, 0.173, 0.867)** ← near-perfect position
- Cube: (0.6-0.8, ±0.2, 0.78)
- Initial distance: 0.05-0.26m ⭐ very close!

Expected output (`debug_orientation.py`):
- **Downward alignment: ~0.805** (1.0 = perfect, 0.805 = excellent!)
- Z-axis direction: (0.067, 0.590, **-0.805**) ← strong negative Z component
- **Joint angles: [0.0, -0.10, 1.80, -2.50, -1.40, 0.0]**

Expected output (`check_arm_horizontal.py`):
- Gripper downward: 0.805 ✅ mostly vertical (good)
- Arm Z variation: ~0.841m ⚠️  (inherent to H2017 kinematics)
- Horizontal ratio: ~0.795 (good enough for task)

## Implementation Notes & Limitations

- **Training Dependency Issue**: OpenSSL/boto3 conflict prevents training script from running. Evaluation works correctly. Training requires fixing the Python environment or using conda.
- TensorBoard logging is disabled by default to avoid bloated dependencies; re-enable by removing the stub logic in the scripts if you need summaries.
- The evaluation CLI uses `DummyVecEnv` without an explicit `Monitor` wrapper, so reward/length stats are taken directly from the VecNormalize wrapper.
- Success criteria focus strictly on alignment; grasping/lifting would require extended reward shaping and contact modeling.
- **Home pose optimized for H2017 kinematics**: `[0.0, -0.10, 1.80, -2.50, -1.40, 0.0]`
  - Multi-stage optimization: position → balanced → **maximum vertical**
  - Result: **0.805 downward alignment** (near-vertical gripper), 0.662m X, 0.867m Z
  - Position accuracy: X error 0.012m, Z error 0.013m
  - Tools: `scripts/maximize_vertical_gripper.py`, `scripts/check_arm_horizontal.py`
- **Trade-off**: H2017's link geometry causes inherent arm tilt (~0.84m Z variation). Gripper verticality prioritized over arm horizontality.

## Acknowledgements

- Doosan robot & gripper MJCFs sourced from the `dsr_mujoco` ROS2 package.
- Stable-Baselines3 (v2.0.0) provides the PPO implementation.

Feel free to iterate on rewards, spawn ranges, or policy architecture to push alignment performance further.
