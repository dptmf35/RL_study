# Unitree Go2 Locomotion RL

MuJoCo 시뮬레이션 환경에서 Unitree Go2 사족보행 로봇의 보행 정책을 강화학습(PPO)으로 학습하고, 학습된 정책을 MuJoCo 뷰어로 시각화 검증하는 프로젝트입니다.

## 프로젝트 구조

```
Locomotion_RL/
├── envs/
│   ├── __init__.py
│   └── go2_env.py          # Go2 Gymnasium 환경 (핵심)
├── configs/
│   └── go2_config.py        # 학습 하이퍼파라미터 설정
├── train.py                 # PPO 학습 스크립트
├── eval.py                  # 정책 평가 및 시각화
├── checkpoints/             # 학습된 모델 저장
├── logs/                    # TensorBoard 로그
└── README.md
```

## 환경 설계

### 로봇 모델
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)의 Unitree Go2 모델을 사용합니다.

- **관절 구성**: 4다리 × 3관절 = 12 액추에이터
  - Hip (abduction): ±23.7 Nm, 범위 [-1.05, 1.05] rad
  - Thigh (hip): ±23.7 Nm, 범위 [-1.57, 3.49] rad (front) / [-0.52, 4.54] rad (back)
  - Calf (knee): ±45.43 Nm, 범위 [-2.72, -0.84] rad
- **기본 자세**: 각 다리 [0, 0.9, -1.8] rad (서있는 자세)
- **로봇 무게**: ~13.7 kg (베이스 6.9kg + 다리)

### 관측 공간 (45차원)
| 관측값 | 차원 | 설명 |
|--------|------|------|
| 중력 벡터 (body frame) | 3 | 몸체 기울기 감지 |
| 속도 명령 (vx, vy, yaw) | 3 | 목표 이동 방향 |
| 관절 위치 (기본 대비) | 12 | 현재 관절 각도 |
| 관절 속도 (×0.05) | 12 | 스케일링된 관절 속도 |
| 이전 행동 | 12 | 행동 smoothness 유도 |
| 몸체 각속도 (×0.25) | 3 | 스케일링된 회전 속도 |

### 행동 공간 (12차원)
- 기본 서있는 자세 대비 관절 위치 오프셋 [-1, 1]
- PD 제어기로 토크 변환: `τ = Kp × (target - current) - Kd × velocity`
- Kp = 40.0, Kd = 1.0, action_scale = 0.25

### 보상 함수
| 보상 항목 | 가중치 | 설명 |
|-----------|--------|------|
| 전방 속도 추적 | +1.0 | exp(-4 × error²) 형태 |
| 횡방향 속도 추적 | +0.5 | exp(-4 × error²) 형태 |
| Yaw rate 추적 | +0.5 | exp(-4 × error²) 형태 |
| 생존 보너스 | +0.5 | 매 스텝 고정 보상 |
| 발 공중 시간 | +1.0 | 교대 보행 패턴 유도 |
| 토크 패널티 | -1e-5 | 에너지 효율 |
| 관절 가속도 패널티 | -2.5e-7 | 부드러운 움직임 |
| 행동 변화율 패널티 | -0.01 | 행동 smoothness |
| 자세 패널티 | -1.0 | 똑바로 서기 유도 |
| 높이 패널티 | -10.0 | 목표 높이(0.27m) 유지 |

### 종료 조건
- 몸체 기울기 > 0.8 rad (넘어짐)
- 몸체 높이 < 0.15m (주저앉음)
- 1000 스텝 (20초) 초과 (truncation)

## 의존성

```bash
pip install mujoco gymnasium stable-baselines3 numpy torch tensorboard
```

| 패키지 | 버전 |
|--------|------|
| MuJoCo | ≥ 3.1 |
| Gymnasium | ≥ 0.28 |
| Stable-Baselines3 | ≥ 2.0 |
| PyTorch | ≥ 2.0 (CUDA 권장) |

## 사용법

### 1. 학습

```bash
cd Locomotion_RL

# 기본 학습 (5M steps, 4 병렬 환경)
python train.py

# 스텝 수 지정
python train.py --timesteps 2000000

# 병렬 환경 수 변경
python train.py --n_envs 8

# 학습 재개
python train.py --resume checkpoints/<run_name>/best_model.zip
```

학습 중 TensorBoard로 모니터링:
```bash
tensorboard --logdir logs/
```

### 2. 평가 (MuJoCo 뷰어)

```bash
# 인터랙티브 뷰어 실행 (학습된 정책 시각화)
python eval.py --model checkpoints/<run_name>/best_model.zip

# 속도 명령 지정
python eval.py --model checkpoints/<run_name>/best_model.zip --cmd_vx 0.8

# 헤드리스 통계 평가
python eval.py --model checkpoints/<run_name>/best_model.zip --no-viewer --episodes 20

# 영상 녹화 (mediapy 필요)
python eval.py --model checkpoints/<run_name>/best_model.zip --record output.mp4
```

### 3. 빠른 테스트 (50k steps)

```bash
python train.py --timesteps 50000 --n_envs 2
```

## PPO 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| learning_rate | 3e-4 | Adam 학습률 |
| n_steps | 2048 | 업데이트당 수집 스텝 |
| batch_size | 64 | 미니배치 크기 |
| n_epochs | 10 | 에폭 수 |
| gamma | 0.99 | 할인 인자 |
| gae_lambda | 0.95 | GAE λ |
| clip_range | 0.2 | PPO 클리핑 범위 |
| ent_coef | 0.01 | 엔트로피 보너스 |
| net_arch | [256, 256, 128] | MLP 은닉층 |

## 설계 원칙

1. **PD 위치 제어**: 직접 토크 대신 목표 관절 위치를 PD 제어기로 추종하여 안정적인 학습
2. **관측 정규화**: VecNormalize로 관측과 보상을 실시간 정규화
3. **Body frame 관측**: 모든 속도/중력 정보를 몸체 좌표계로 변환하여 sim-to-real 친화적
4. **Exponential 보상**: 속도 추적에 exp(-error²) 형태로 sparse reward 문제 완화
5. **교대 보행 유도**: feet air time 보상으로 자연스러운 걸음걸이 학습
