# UR5e Pick and Place - MuJoCo Reinforcement Learning

MuJoCo 기반의 Universal Robots UR5e 로봇 팔을 사용한 Pick and Place 강화학습 프로젝트입니다. PPO (Proximal Policy Optimization) 알고리즘을 사용하여 로봇이 무작위로 배치된 큐브를 집어 목표 위치에 놓는 작업을 학습합니다.

## 프로젝트 구조

```
RL_study/
├── assets/
│   ├── ur5e/                    # UR5e 로봇 메시 파일
│   │   ├── assets/              # OBJ 메시 파일들
│   │   ├── ur5e.xml             # 원본 UR5e 모델
│   │   └── scene.xml            # 기본 씬
│   └── ur5e_pick_place.xml      # Pick & Place 환경 XML
├── configs/
│   └── default.yaml             # 기본 설정 파일
├── envs/
│   ├── __init__.py
│   └── ur5e_pick_place_env.py   # Gymnasium 환경 구현
├── scripts/
│   ├── train_ppo.py             # PPO 학습 스크립트
│   ├── evaluate.py              # 평가 및 시각화 스크립트
│   └── record_video.py          # 비디오 녹화 스크립트
├── logs/                        # TensorBoard 로그 (학습 후 생성)
├── models/                      # 학습된 모델 (학습 후 생성)
├── videos/                      # 녹화된 비디오 (녹화 후 생성)
├── requirements.txt             # Python 의존성
└── README.md                    # 프로젝트 문서
```

## 환경 설명

### 시뮬레이션 환경
- **로봇**: Universal Robots UR5e (6 DOF 로봇 팔)
- **그리퍼**: 평행 조 그리퍼 (2 fingers)
- **객체**: 빨간색 큐브 (5cm x 5cm x 5cm)
- **목표**: 녹색 원형 영역에 큐브 배치
- **테이블**: 작업 공간을 제공하는 테이블

### 관측 공간 (Observation Space)
32차원 연속 벡터:
- 관절 위치 (6)
- 관절 속도 (6)
- 그리퍼 위치 (2)
- End-effector 위치 (3)
- 큐브 위치 (3)
- 큐브 속도 (3)
- 목표 위치 (3)
- 큐브-그리퍼 상대 위치 (3)
- 큐브-목표 상대 위치 (3)

### 행동 공간 (Action Space)
7차원 연속 벡터 [-1, 1]:
- 관절 위치 변화량 (6) - 각 관절의 delta 위치
- 그리퍼 명령 (1) - 0: 열기, 1: 닫기

### 보상 함수 (Reward Function)
**Dense Reward** (기본값):
- **Reach**: 그리퍼와 큐브 사이 거리에 비례하는 음의 보상
- **Grasp**: 큐브 근처에서 그리퍼를 닫으면 +2.0
- **Lift**: 큐브를 들어올리면 +5.0
- **Place**: 목표 위치와의 거리에 비례하는 보상
- **Success**: 작업 완료 시 +100.0

**Sparse Reward**:
- 작업 완료 시에만 +100.0

## 설치 방법

### 1. Python 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. MuJoCo 설치 확인
MuJoCo 3.0 이상이 설치되어 있어야 합니다. `mujoco` 패키지가 자동으로 설치됩니다.

## 사용 방법

### 환경 테스트
```bash
# 랜덤 액션으로 환경 테스트
python scripts/evaluate.py --random --n-episodes 3
```

### PPO 학습
```bash
# 기본 설정으로 학습
python scripts/train_ppo.py

# 커스텀 설정으로 학습
python scripts/train_ppo.py \
    --total-timesteps 5000000 \
    --n-envs 16 \
    --learning-rate 1e-4 \
    --batch-size 512

# TensorBoard로 학습 모니터링
tensorboard --logdir logs/
```

### 학습된 모델 평가
```bash
# 학습된 모델로 평가
python scripts/evaluate.py \
    --model-path models/<run_name>/best/best_model.zip \
    --n-episodes 20

# 슬로우 모션으로 시각화
python scripts/evaluate.py \
    --model-path models/<run_name>/final_model.zip \
    --slow-motion
```

### 비디오 녹화
```bash
# 학습된 모델의 에피소드 녹화
python scripts/record_video.py \
    --model-path models/<run_name>/best/best_model.zip \
    --output-dir videos \
    --n-episodes 5

# 랜덤 액션 녹화
python scripts/record_video.py --n-episodes 3
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
| `ent_coef` | 0.01 | 엔트로피 계수 |

## 주요 파일 설명

### `envs/ur5e_pick_place_env.py`
Gymnasium 호환 강화학습 환경. MuJoCo 시뮬레이션을 래핑하고 관측, 행동, 보상을 정의합니다.

### `scripts/train_ppo.py`
Stable-Baselines3를 사용한 PPO 학습 스크립트. 병렬 환경, 체크포인트 저장, 평가 콜백 등을 지원합니다.

### `scripts/evaluate.py`
학습된 모델을 로드하여 평가하고 시각화하는 스크립트.

### `assets/ur5e_pick_place.xml`
MuJoCo 시뮬레이션 환경 정의 파일. UR5e 로봇, 그리퍼, 테이블, 큐브, 목표 영역을 정의합니다.

## 예상 학습 결과

- **200만 스텝**: 기본적인 reaching 동작 학습
- **500만 스텝**: 안정적인 grasping 학습
- **1000만 스텝 이상**: 높은 성공률의 pick-and-place 학습

학습 환경과 하이퍼파라미터에 따라 결과가 다를 수 있습니다.

## 커스터마이징

### 새로운 보상 함수 추가
`envs/ur5e_pick_place_env.py`의 `_compute_reward()` 메서드를 수정하세요.

### 환경 수정
`assets/ur5e_pick_place.xml`을 수정하여 객체, 테이블 크기, 목표 위치 등을 변경할 수 있습니다.

### 네트워크 구조 변경
`scripts/train_ppo.py`의 `policy_kwargs`를 수정하여 네트워크 구조를 변경할 수 있습니다.

## 문제 해결

### MuJoCo 렌더링 오류
```bash
# GLFW 설치 (Ubuntu)
sudo apt-get install libglfw3 libglfw3-dev

# 또는 EGL 백엔드 사용
export MUJOCO_GL=egl
```

### CUDA 메모리 부족
`--n-envs` 값을 줄이거나 `--batch-size`를 줄여보세요.

### 학습이 진행되지 않음
- `--reward-type sparse` 대신 `dense`를 사용하세요
- 학습률을 조정해보세요
- 더 많은 timesteps로 학습하세요

## 참고 자료

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [MuJoCo Menagerie (UR5e Model)](https://github.com/google-deepmind/mujoco_menagerie)

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
