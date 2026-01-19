# Custom ManiSkill Environment with PPO Training

PANDA 로봇으로 큐브 집기 태스크를 PPO로 학습하는 예제입니다.

## 파일 구조

```
stable_baseline/
├── my_pick_cube_env.py        # 커스텀 환경 정의
├── train_ppo_pickcube.py      # PPO 학습 스크립트
├── eval_ppo_pickcube.py       # 모델 평가 스크립트
└── README.md                   # 이 파일
```

## 환경 설명

**MyPickCube-v0**
- **로봇**: PANDA (7-DOF arm + 2-finger gripper)
- **태스크**: 테이블 위 빨간 큐브를 집어서 목표 높이(0.3m)까지 들어올리기
- **관측**: `state` 모드 (로봇 joint pos/vel, 큐브 pose/vel, TCP-큐브 거리 등)
- **액션**: `pd_joint_delta_pos` (관절 위치 delta 제어)
- **리워드**:
  - Reaching: TCP와 큐브 사이 거리
  - Grasping: 큐브를 잡고 있는지
  - Lifting: 큐브 높이
  - Success: 목표 높이 도달 시 보너스

## 필요한 패키지

```bash
pip install mani-skill stable-baselines3 gymnasium
```

## 사용 방법

### 1) 학습

```bash
cd /home/yeseul/Desktop/mygitrepos/RL_study/stable_baseline
python train_ppo_pickcube.py
```

**학습 설정:**
- 병렬 환경: 32개
- Total timesteps: 500,000
- 평가 주기: 25,000 steps마다
- 체크포인트: 50,000 steps마다

**결과물:**
- `ppo_pickcube_final.zip` - 최종 모델
- `best_model/` - 평가에서 가장 좋은 성능의 모델
- `checkpoints/` - 중간 체크포인트들
- `eval_videos/` - 평가 영상
- `tensorboard_logs/` - Tensorboard 로그

### 2) Tensorboard로 학습 모니터링

```bash
tensorboard --logdir=tensorboard_logs/
```

브라우저에서 `http://localhost:6006` 접속

### 3) 학습된 모델 평가

```bash
# 최종 모델 평가 (10 에피소드, 영상 저장)
python eval_ppo_pickcube.py --model ppo_pickcube_final --episodes 10

# Best 모델 평가
python eval_ppo_pickcube.py --model best_model/best_model --episodes 20

# 영상 저장 안하고 빠르게 평가
python eval_ppo_pickcube.py --model ppo_pickcube_final --episodes 50 --no-render
```

### 4) 환경만 테스트 (랜덤 행동)

```python
import gymnasium as gym
import my_pick_cube_env

env = gym.make("MyPickCube-v0", num_envs=1, obs_mode="state", render_mode="human")
obs, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated.any() or truncated.any():
        obs, info = env.reset()

env.close()
```

## 커스터마이징

### 리워드 함수 수정
`my_pick_cube_env.py`의 `compute_dense_reward()` 메서드 수정

### 난이도 조절
- `goal_height`: 목표 높이 변경 (현재 0.3m)
- `max_episode_steps`: 에피소드 길이 변경 (현재 200)
- 큐브 스폰 위치 랜덤화 범위 조절

### PPO 하이퍼파라미터 튜닝
`train_ppo_pickcube.py`에서:
- `learning_rate`: 학습률
- `n_steps`: 업데이트당 스텝 수
- `batch_size`: 배치 크기
- `gamma`: 할인율
- `gae_lambda`: GAE lambda

## 다음 단계

### 카메라 관측으로 확장
1. `obs_mode="sensor_data"` 또는 `"rgbd"` 사용
2. PPO policy를 `"CnnPolicy"`로 변경
3. 필요시 feature extractor 커스터마이징

### 다른 로봇 사용
1. `SUPPORTED_ROBOTS`에 추가 (예: `"fetch"`, `"xarm7"`)
2. `robot_uids` 파라미터로 선택

### 더 복잡한 태스크
1. 여러 오브젝트 추가
2. 장애물 추가
3. 두 팔 조작 (dual-arm)
4. Scene manipulation (ReplicaCAD 등)

## 문제 해결

### GPU 메모리 부족
- `num_envs` 줄이기 (32 → 16 or 8)
- `sim_backend="cpu"` 사용

### 학습이 안 됨
- 리워드 스케일 확인 (normalized reward 사용 권장)
- 초기 성공 샘플이 있는지 확인
- Curriculum learning 고려

### 렌더링 오류
- `render_mode=None`으로 학습 (평가만 렌더링)
- GPU 드라이버 확인
- `sim_backend="cpu"` 시도
