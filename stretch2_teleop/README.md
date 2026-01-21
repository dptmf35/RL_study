# Stretch Robot Grasping - Reinforcement Learning

Hello Robot Stretch 2 로봇의 매니퓰레이터로 큐브를 잡는 강화학습 프로젝트입니다.

## 프로젝트 구조

```
stretch2_teleop/
├── assets/
│   ├── stretch.xml              # 원본 Stretch 모델
│   ├── scene.xml                # 기본 씬
│   ├── stretch_fixed_base.xml   # 고정 베이스 학습 환경
│   └── assets/                  # 메시 파일들
├── envs/
│   ├── __init__.py
│   └── stretch_grasp_env.py     # Gymnasium 환경
├── scripts/
│   ├── train_sac.py             # SAC 학습 스크립트
│   └── evaluate.py              # 평가/시각화 스크립트
├── teleop_stretch.py            # 텔레오퍼레이션 스크립트
├── models/                      # 학습된 모델
├── logs/                        # TensorBoard 로그
└── README.md
```

## 환경 설명

### 태스크
- **목표**: 테이블 위의 큐브를 잡아서 들어올리기
- **로봇**: Hello Robot Stretch 2 (베이스 고정)
- **행동 공간**: 4차원 연속 (lift, arm_extend, wrist_yaw, gripper)
- **관측 공간**: 16차원 (관절 상태, 위치, 속도 등)

### 행동 공간 (Action Space)
| Index | Action | Range | 설명 |
|-------|--------|-------|------|
| 0 | lift | [-1, 1] | 수직 이동 |
| 1 | arm_extend | [-1, 1] | 팔 확장/수축 |
| 2 | wrist_yaw | [-1, 1] | 손목 회전 |
| 3 | gripper | [-1, 1] | 그리퍼 (-1: 열기, 1: 닫기) |

### 보상 함수 (Dense Reward)
- **Reach**: 그리퍼가 큐브에 가까울수록 높은 보상
- **Grasp**: 큐브 근처에서 +2, 실제 그립 시 +5
- **Lift**: 큐브를 들어올리면 높이에 비례한 보상
- **Success**: 목표 높이 이상 들어올리면 +50

## 사용 방법

### 1. 텔레오퍼레이션 (키보드 조작)
```bash
source /home/yeseul/Desktop/XLeRobot/simulation/Maniskill/.manyvenv/bin/activate
cd /home/yeseul/Desktop/mygitrepos/RL_study/stretch2_teleop
python teleop_stretch.py
```

### 2. 환경 테스트 (랜덤 행동)
```bash
python scripts/evaluate.py --random --n-episodes 3
```

### 3. SAC 학습
```bash
# 기본 학습 (500K steps)
python scripts/train_sac.py

# 더 긴 학습
python scripts/train_sac.py --total-timesteps 2000000

# TensorBoard 모니터링
tensorboard --logdir logs/
```

### 4. 학습된 모델 평가
```bash
python scripts/evaluate.py \
    --model-path models/<run_name>/best/best_model.zip \
    --n-episodes 20

# 슬로우 모션
python scripts/evaluate.py \
    --model-path models/<run_name>/final_model.zip \
    --slow-motion
```

## 알고리즘: SAC (Soft Actor-Critic)

### 선택 이유
- **연속 행동 공간**에 최적화
- **Sample Efficiency** 높음 (off-policy)
- **Entropy Regularization**으로 탐색 장려
- 로봇 매니퓰레이션에서 검증된 성능

### 주요 하이퍼파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| learning_rate | 3e-4 | 학습률 |
| buffer_size | 1M | 리플레이 버퍼 크기 |
| batch_size | 256 | 배치 크기 |
| tau | 0.005 | Soft update 계수 |
| gamma | 0.99 | 할인 계수 |
| ent_coef | auto | 엔트로피 계수 (자동 조정) |

## 학습 팁

1. **학습 시간**: 500K ~ 2M steps 권장
2. **성공률 확인**: TensorBoard에서 `custom/success_rate` 모니터링
3. **큐브 위치 고정**: `--no-randomize` 옵션으로 더 쉬운 학습 시작 가능

## 참고 자료

- [Hello Robot Stretch](https://hello-robot.com/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
