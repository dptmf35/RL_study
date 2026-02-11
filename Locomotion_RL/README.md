# Unitree Go2 Locomotion RL

MuJoCo 시뮬레이션 환경에서 Unitree Go2 사족보행 로봇의 보행 정책을 강화학습(PPO)으로 학습하고, 학습된 정책을 MuJoCo 뷰어로 시각화 검증하는 프로젝트입니다.

**주요 기능:**
- 평지 및 지형(경사, 계단, 거친 지면) 보행 학습
- MuJoCo 뷰어를 통한 실시간 시각화 검증
- 목표 위치 네비게이션 (학습된 정책 + 상위 플래너)

## 프로젝트 구조

```
Locomotion_RL/
├── envs/
│   ├── __init__.py
│   ├── go2_env.py             # Go2 평지 Gymnasium 환경
│   ├── go2_terrain_env.py     # Go2 지형 환경 (heightfield)
│   └── terrain.py             # 절차적 지형 생성기
├── configs/
│   └── go2_config.py          # 학습 하이퍼파라미터 설정
├── train.py                   # PPO 학습 (평지 / 지형)
├── eval.py                    # 정책 평가 및 시각화
├── navigate.py                # 목표 위치 네비게이션
├── checkpoints/               # 학습된 모델 저장
├── logs/                      # TensorBoard 로그
└── README.md
```

## 환경 설계

### 로봇 모델
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)의 Unitree Go2 모델을 사용합니다.

- **관절 구성**: 4다리 x 3관절 = 12 액추에이터
  - Hip (abduction): ±23.7 Nm, 범위 [-1.05, 1.05] rad
  - Thigh (hip): ±23.7 Nm, 범위 [-1.57, 3.49] rad (front) / [-0.52, 4.54] rad (back)
  - Calf (knee): ±45.43 Nm, 범위 [-2.72, -0.84] rad
- **기본 자세**: 각 다리 [0, 0.9, -1.8] rad (서있는 자세)
- **로봇 무게**: ~13.7 kg (베이스 6.9kg + 다리)

### 관측 공간

#### 평지 환경 (45차원)
| 관측값 | 차원 | 설명 |
|--------|------|------|
| 중력 벡터 (body frame) | 3 | 몸체 기울기 감지 |
| 속도 명령 (vx, vy, yaw) | 3 | 목표 이동 방향 |
| 관절 위치 (기본 대비) | 12 | 현재 관절 각도 |
| 관절 속도 (x0.05) | 12 | 스케일링된 관절 속도 |
| 이전 행동 | 12 | 행동 smoothness 유도 |
| 몸체 각속도 (x0.25) | 3 | 스케일링된 회전 속도 |

#### 지형 환경 (53차원) - 평지 관측 + 추가 8차원
| 추가 관측값 | 차원 | 설명 |
|-------------|------|------|
| 발 아래 지형 높이 | 4 | 각 발 위치의 지형 높이 (body 상대) |
| 전방 지형 스캔 | 4 | 0.15~0.60m 전방 지형 높이 |

### 행동 공간 (12차원)
- 기본 서있는 자세 대비 관절 위치 오프셋 [-1, 1]
- PD 제어기로 토크 변환: `τ = Kp × (target - current) - Kd × velocity`
- Kp = 40.0, Kd = 1.0, action_scale = 0.25

### 보상 함수
| 보상 항목 | 가중치 | 설명 |
|-----------|--------|------|
| 전방 속도 추적 | +1.0 | exp(-4 x error²) 형태 |
| 횡방향 속도 추적 | +0.5 | exp(-4 x error²) 형태 |
| Yaw rate 추적 | +0.5 | exp(-4 x error²) 형태 |
| 생존 보너스 | +0.5 | 매 스텝 고정 보상 |
| 발 공중 시간 | +1.0 | 교대 보행 패턴 유도 |
| 토크 패널티 | -1e-5 | 에너지 효율 |
| 관절 가속도 패널티 | -2.5e-7 | 부드러운 움직임 |
| 행동 변화율 패널티 | -0.01 | 행동 smoothness |
| 자세 패널티 | -1.0 | 똑바로 서기 유도 |
| 높이 패널티 | -10.0 | 목표 높이 유지 (지형 상대적) |

### 지형 생성기

`envs/terrain.py`에서 절차적으로 heightfield 지형을 생성합니다.

| 지형 유형 | 설명 |
|-----------|------|
| flat | 평탄한 지면 (약간의 높이 변화) |
| slope | 경사면 (오르막/내리막) |
| stairs | 계단 (단차 높이는 난이도에 비례) |
| rough | 불규칙한 거친 지면 |

- **지형 크기**: 20m x 20m (200x200 heightfield, 0.1m 해상도)
- **난이도**: 0.0 (평지) ~ 1.0 (험한 지형)
- **배치**: 로봇 시작점(x≈0) 주변은 평탄, x > 1m 부터 지형 시작
- **패치 기반**: 2.5m 단위 패치로 다양한 지형 타입 혼합

### 종료 조건
- 몸체 기울기 > 0.8 rad (넘어짐)
- 몸체 높이 < 0.15m / 지형 위 0.10m 미만 (주저앉음)
- 1000 스텝 (20초) 초과 (truncation)

## 의존성

```bash
pip install mujoco gymnasium stable-baselines3 numpy torch tensorboard
```

| 패키지 | 버전 |
|--------|------|
| MuJoCo | >= 3.1 |
| Gymnasium | >= 0.28 |
| Stable-Baselines3 | >= 2.0 |
| PyTorch | >= 2.0 (CUDA 권장) |

## 사용법

### 1. 평지 보행 학습

```bash
cd Locomotion_RL

# 기본 학습 (5M steps, 4 병렬 환경)
python train.py

# 스텝 수 / 병렬 환경 수 지정
python train.py --timesteps 2000000 --n_envs 8

# 학습 재개
python train.py --resume checkpoints/<run_name>/best_model.zip
```

### 2. 지형 보행 학습

```bash
# 기본 지형 학습 (난이도 0.5)
python train.py --terrain

# 난이도 조절 (0.0=평지, 1.0=험한 지형)
python train.py --terrain --difficulty 0.3     # 쉬운 지형
python train.py --terrain --difficulty 0.8     # 어려운 지형

# 빠른 테스트
python train.py --terrain --timesteps 50000 --n_envs 2
```

학습 중 TensorBoard로 모니터링:
```bash
tensorboard --logdir logs/
```

### 3. 평가 (MuJoCo 뷰어)

```bash
# 평지 정책 시각화
python eval.py --model checkpoints/<run_name>/best_model.zip

# 지형 정책 시각화
python eval.py --model checkpoints/<run_name>/best_model.zip --terrain

# 속도 명령 지정
python eval.py --model checkpoints/<run_name>/best_model.zip --cmd_vx 0.8

# 헤드리스 통계 평가
python eval.py --model checkpoints/<run_name>/best_model.zip --no-viewer --episodes 20

# 영상 녹화 (pip install mediapy)
python eval.py --model checkpoints/<run_name>/best_model.zip --record output.mp4
```

### 4. 목표 위치 네비게이션

학습된 보행 정책을 사용해 원하는 위치로 로봇을 이동시킵니다.

```bash
# 평지 네비게이션
python navigate.py --model checkpoints/<run_name>/best_model.zip

# 지형 위 네비게이션
python navigate.py --model checkpoints/<run_name>/best_model.zip --terrain --difficulty 0.3
```

**네비게이션 사용 방법:**
1. 실행하면 MuJoCo 뷰어가 열리고 터미널에 입력 프롬프트가 표시됩니다
2. 터미널에서 목표 좌표를 입력합니다 (예: `3 2` → x=3m, y=2m)
3. 로봇이 목표 위치(빨간 구 마커)를 향해 걸어갑니다
4. 도착하면 로봇이 자동으로 정지합니다
5. 새 목표를 입력하면 다시 이동합니다

| 명령 | 설명 |
|------|------|
| `x y` | 목표 위치 설정 (예: `3 2`) |
| `stop` | 목표 해제 (제자리 정지) |
| `quit` | 종료 |

### 네비게이션 아키텍처

```
[목표 위치 (x, y)]
        |
+-------------------+
|  Navigation       |  목표 방향/거리 → 속도 명령 변환
|  Planner          |  (비례 제어, 도착 감지)
+-------------------+
        | (vx, vy, yaw_rate)
+-------------------+
|  Locomotion       |  속도 명령 추종 보행
|  Policy (PPO)     |  (학습된 정책, 그대로 사용)
+-------------------+
        | (joint torques)
      [Robot]
```

- **상위 레벨**: 목표까지의 거리/방향을 계산하여 속도 명령(vx, vy, yaw_rate) 생성
- **하위 레벨**: 학습된 보행 정책이 속도 명령을 추종하여 실제 보행 수행
- 보행 정책은 **재학습 없이 그대로 재사용**됩니다

## PPO 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| learning_rate | 3e-4 | Adam 학습률 |
| n_steps | 2048 | 업데이트당 수집 스텝 |
| batch_size | 64 | 미니배치 크기 |
| n_epochs | 10 | 에폭 수 |
| gamma | 0.99 | 할인 인자 |
| gae_lambda | 0.95 | GAE lambda |
| clip_range | 0.2 | PPO 클리핑 범위 |
| ent_coef | 0.01 | 엔트로피 보너스 |
| net_arch | [256, 256, 128] | MLP 은닉층 |

## 설계 원칙

1. **PD 위치 제어**: 직접 토크 대신 목표 관절 위치를 PD 제어기로 추종하여 안정적인 학습
2. **관측 정규화**: VecNormalize로 관측과 보상을 실시간 정규화
3. **Body frame 관측**: 모든 속도/중력 정보를 몸체 좌표계로 변환하여 sim-to-real 친화적
4. **Exponential 보상**: 속도 추적에 exp(-error²) 형태로 sparse reward 문제 완화
5. **교대 보행 유도**: feet air time 보상으로 자연스러운 걸음걸이 학습
6. **지형 상대 보상**: 지형 환경에서 높이/종료 조건이 지형 표면 기준으로 동작
7. **계층적 제어**: 보행 정책(저수준) + 네비게이션 플래너(고수준) 분리로 재사용성 확보
