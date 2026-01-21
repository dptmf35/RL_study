# UR5e + Robotiq 2F85 Pick and Place - MuJoCo Reinforcement Learning

MuJoCo 기반의 Universal Robots UR5e 로봇 팔과 Robotiq 2F85 그리퍼를 사용한 Pick and Place 강화학습 프로젝트입니다. PPO (Proximal Policy Optimization) 알고리즘을 사용하여 로봇이 무작위로 배치된 큐브를 집어 목표 위치에 놓는 작업을 학습합니다.

## 프로젝트 구조

```
RL_study/
├── assets/
│   └── ur5e_robotiq_pick_place.xml  # UR5e + Robotiq 2F85 환경
├── envs/
│   ├── __init__.py
│   └── ur5e_robotiq_pick_place_env.py  # Gymnasium 환경 구현
├── scripts/
│   ├── train_ppo_robotiq.py        # PPO 학습 스크립트
│   └── evaluate_robotiq.py         # 평가 및 시각화 스크립트
├── mujoco_menagerie/
│   ├── universal_robots_ur5e/      # UR5e 로봇 모델
│   └── robotiq_2f85/               # Robotiq 2F85 그리퍼 모델
├── logs/                           # TensorBoard 로그
├── models/                         # 학습된 모델
└── README_ROBOTIQ.md              # 이 문서
```

## 환경 설명

### 시뮬레이션 환경
- **로봇**: Universal Robots UR5e (6 DOF 로봇 팔)
- **그리퍼**: Robotiq 2F85 (실제 4-bar linkage 메커니즘)
- **객체**: 빨간색 큐브 (3cm x 3cm x 3cm)
- **목표**: 녹색 원형 영역에 큐브 배치
- **테이블**: 작업 공간을 제공하는 테이블 (0.6m x 0.6m)

### Robotiq 2F85 그리퍼 특징
- **실제 물리**: 4-bar linkage 메커니즘으로 정밀한 파지 구현
- **제어 범위**: 0-255 (0=완전 열림, 255=완전 닫힘)
- **최대 개구부**: ~8.5cm
- **평행 그립**: 양쪽 손가락이 평행하게 움직임
- **스프링**: 스프링 링크로 안정적인 파지

### 관측 공간 (Observation Space)
32차원 연속 벡터:
- 관절 위치 (6)
- 관절 속도 (6)
- 그리퍼 위치 (1) - driver joint position
- 그리퍼 속도 (1)
- End-effector 위치 (3) - pinch site
- 큐브 위치 (3)
- 큐브 속도 (3)
- 목표 위치 (3)
- 큐브-그리퍼 상대 위치 (3)
- 큐브-목표 상대 위치 (3)

### 행동 공간 (Action Space)
7차원 연속 벡터 [-1, 1]:
- 관절 위치 변화량 (6) - 각 관절의 delta 위치
- 그리퍼 명령 (1) - -1: 열기, 1: 닫기 (내부적으로 0-255로 변환)

### 보상 함수 (Reward Function)
**Dense Reward** (기본값):
- **Reach**: 그리퍼와 큐브 사이 거리 기반 단계별 보상
- **Gripper Prep**: 큐브 근처에서 그리퍼를 여는 동작 (+3.0)
- **Grasp**: 올바른 순서로 큐브를 잡으면 (+5.0 ~ +8.0)
- **Lift**: 큐브를 들어올리면 (최대 +10.0)
- **Place**: 목표 위치와의 거리에 비례하는 보상 (최대 +5.0)
- **Success**: 작업 완료 시 (+100.0)

**Sparse Reward**:
- 작업 완료 시에만 +100.0

## 설치 방법

### 1. Python 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. MuJoCo Menagerie 모델 확인
프로젝트에는 다음 모델이 포함되어 있습니다:
- `mujoco_menagerie/universal_robots_ur5e/` - UR5e 로봇
- `mujoco_menagerie/robotiq_2f85/` - Robotiq 2F85 그리퍼

## 사용 방법

### 환경 테스트
```bash
# 간단한 로드 테스트
python test_robotiq_env.py

# 랜덤 액션으로 환경 테스트
python scripts/evaluate_robotiq.py --random --n-episodes 3

# 정적 데모 (로봇이 움직이지 않음)
python scripts/evaluate_robotiq.py --static --n-episodes 1
```

### PPO 학습

#### 기본 학습
```bash
# Full task (pick and place)
python scripts/train_ppo_robotiq.py \
    --total-timesteps 2000000 \
    --n-envs 8

# Easy mode (큐브가 가까운 곳에 spawn)
python scripts/train_ppo_robotiq.py \
    --easy-mode \
    --total-timesteps 1000000 \
    --n-envs 8
```

#### 단계별 학습 (권장)
```bash
# Step 1: Reach task (50만 스텝)
python scripts/train_ppo_robotiq.py \
    --task-mode reach \
    --easy-mode \
    --total-timesteps 500000 \
    --n-envs 8

# Step 2: Pick task (100만 스텝)
python scripts/train_ppo_robotiq.py \
    --task-mode pick \
    --easy-mode \
    --total-timesteps 1000000 \
    --n-envs 8

# Step 3: Full pick_place task (200만 스텝)
python scripts/train_ppo_robotiq.py \
    --task-mode pick_place \
    --total-timesteps 2000000 \
    --n-envs 8
```

#### 고급 설정
```bash
python scripts/train_ppo_robotiq.py \
    --total-timesteps 5000000 \
    --n-envs 16 \
    --learning-rate 3e-4 \
    --batch-size 512 \
    --ent-coef 0.02 \
    --device cuda

# TensorBoard로 학습 모니터링
tensorboard --logdir logs/
```

### 학습된 모델 평가

```bash
# 학습된 모델로 평가
python scripts/evaluate_robotiq.py \
    --model-path models/<run_name>/best/best_model.zip \
    --n-episodes 20

# Deterministic 평가
python scripts/evaluate_robotiq.py \
    --model-path models/<run_name>/final_model.zip \
    --deterministic \
    --n-episodes 50

# 슬로우 모션으로 시각화
python scripts/evaluate_robotiq.py \
    --model-path models/<run_name>/best/best_model.zip \
    --slow-motion \
    --n-episodes 5

# Easy mode로 평가
python scripts/evaluate_robotiq.py \
    --model-path models/<run_name>/best/best_model.zip \
    --easy-mode \
    --n-episodes 20
```

## 학습 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `total_timesteps` | 2,000,000 | 총 학습 스텝 |
| `n_envs` | 8 | 병렬 환경 수 |
| `learning_rate` | 3e-4 | 학습률 |
| `n_steps` | 2048 | 업데이트당 스텝 |
| `batch_size` | 256 | 미니배치 크기 |
| `n_epochs` | 10 | 업데이트당 에폭 |
| `gamma` | 0.99 | 할인 계수 |
| `gae_lambda` | 0.95 | GAE 람다 |
| `clip_range` | 0.2 | PPO 클립 범위 |
| `ent_coef` | 0.02 | 엔트로피 계수 |

## 주요 파일 설명

### `assets/ur5e_robotiq_pick_place.xml`
UR5e 로봇과 Robotiq 2F85 그리퍼가 통합된 MuJoCo 환경 정의 파일.
- Mujoco Menagerie의 공식 모델 사용
- 실제 물리 기반 4-bar linkage 그리퍼
- 테이블, 큐브, 목표 영역 포함

### `envs/ur5e_robotiq_pick_place_env.py`
Gymnasium 호환 강화학습 환경.
- 32차원 관측 공간
- 7차원 행동 공간 (6 arm + 1 gripper)
- Dense/Sparse 보상 함수
- Task mode 지원 (reach/pick/pick_place)

### `scripts/train_ppo_robotiq.py`
Stable-Baselines3 기반 PPO 학습 스크립트.
- 병렬 환경 지원
- VecNormalize 적용
- 자동 체크포인트 및 평가

### `scripts/evaluate_robotiq.py`
학습된 모델 평가 및 데모 스크립트.
- 모델 평가 모드
- 랜덤 액션 데모
- 정적 로봇 데모

## 예상 학습 결과

### Task Mode별 학습 시간
- **Reach** (50만 스텝): 큐브에 도달하는 기본 동작 학습
- **Pick** (100만 스텝): 안정적인 grasping 학습
- **Pick & Place** (200만 스텝 이상): 높은 성공률의 full task 학습

### Robotiq 그리퍼 특징에 따른 학습 난이도
- **장점**: 실제 물리 기반으로 더 안정적인 파지
- **단점**: 복잡한 4-bar linkage로 인해 학습 초기 탐색이 어려움
- **권장**: Easy mode로 시작하여 점진적으로 난이도 증가

## 커스터마이징

### 그리퍼 제어 방식 변경
`envs/ur5e_robotiq_pick_place_env.py`의 `step()` 메서드에서 그리퍼 제어 매핑 수정:
```python
# 현재: [-1, 1] -> [0, 255]
gripper_ctrl = (gripper_action + 1.0) * 127.5

# 더 세밀한 제어를 원한다면:
gripper_ctrl = np.interp(gripper_action, [-1, 1], [0, 255])
```

### 큐브 크기 변경
`assets/ur5e_robotiq_pick_place.xml`에서:
```xml
<!-- 현재: 3cm 큐브 -->
<geom name="cube_geom" type="box" size="0.015 0.015 0.015" .../>

<!-- 더 작은 큐브 (2cm): -->
<geom name="cube_geom" type="box" size="0.01 0.01 0.01" .../>
```

### 보상 함수 수정
`envs/ur5e_robotiq_pick_place_env.py`의 `_compute_reward()` 메서드 수정.

## 문제 해결

### 그리퍼가 큐브를 잡지 못함
1. Easy mode로 시작: `--easy-mode`
2. Reach task부터 단계적 학습
3. 그리퍼 마찰력 증가 (XML의 `friction` 파라미터)

### 학습이 진행되지 않음
- `--ent-coef 0.02` 이상으로 설정 (탐색 증가)
- `--learning-rate 1e-4`로 감소
- Easy mode에서 먼저 학습

### MuJoCo 렌더링 오류
```bash
# Ubuntu
sudo apt-get install libglfw3 libglfw3-dev

# 또는 EGL 백엔드 사용
export MUJOCO_GL=egl
```

### 충돌 감지 문제
XML 파일의 contact 설정 확인:
- `solref`: 충돌 강도 (작을수록 강함)
- `solimp`: 침투 방지 파라미터

## 기술적 세부사항

### Robotiq 2F85 제어
- **Actuator**: `fingers_actuator` (tendon 기반)
- **제어 범위**: 0-255
- **Driver joints**: `right_driver_joint`, `left_driver_joint`
- **Equality constraints**: 양쪽 손가락 동기화

### 충돌 감지
```xml
<contact>
  <pair geom1="table_top" geom2="left_pad1" solref="0.02 1"/>
  <pair geom1="cube_geom" geom2="left_pad1" friction="1.5 0.5 ..."/>
</contact>
```

### 관측 사이트
- `pinch`: 그리퍼 중심 (Z=0.145m from gripper base)
- `target_site`: 목표 위치
- `cube_site`: 큐브 중심

## 성능 벤치마크

### 하드웨어 요구사항
- **CPU**: 최소 8코어 (병렬 환경 실행)
- **RAM**: 16GB 이상
- **GPU**: CUDA 지원 GPU (선택사항, 학습 속도 향상)

### 학습 시간 (예상)
- **8 parallel envs, CPU**: ~6-8시간 (200만 스텝)
- **16 parallel envs, GPU**: ~3-4시간 (200만 스텝)

## 참고 자료

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [MuJoCo Menagerie (UR5e Model)](https://github.com/google-deepmind/mujoco_menagerie)
- [Robotiq 2F-85 Specifications](https://robotiq.com/products/2f85-140-adaptive-robot-gripper)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 제안, Pull Request는 환영합니다!
