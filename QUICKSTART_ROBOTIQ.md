# UR5e + Robotiq 2F85 퀵스타트 가이드

Robotiq 2F85 그리퍼가 장착된 UR5e 로봇으로 강화학습을 빠르게 시작하는 가이드입니다.

## 1. 환경 테스트 (5분)

### 환경 로드 확인
```bash
cd ~/Desktop/mygitrepos/RL_study

# 가상환경 활성화
source /home/yeseul/Desktop/XLeRobot/simulation/Maniskill/.manyvenv/bin/activate

# 환경 테스트
python3 test_robotiq_env.py
```

### 랜덤 액션 데모
```bash
python3 scripts/evaluate_robotiq.py --random --n-episodes 3
```

## 2. 첫 번째 학습 (Easy Mode)

### Step 1: Reach 학습 (30분)
```bash
python3 scripts/train_ppo_robotiq.py \
    --task-mode reach \
    --easy-mode \
    --total-timesteps 500000 \
    --n-envs 8
```

이 단계에서 로봇은:
- ✓ 큐브를 향해 손을 뻗는 것을 학습
- ✓ 장애물을 피하는 것을 학습
- ✓ 큐브에 도달하면 +20 보상

### Step 2: Pick 학습 (1시간)
```bash
python3 scripts/train_ppo_robotiq.py \
    --task-mode pick \
    --easy-mode \
    --total-timesteps 1000000 \
    --n-envs 8
```

이 단계에서 로봇은:
- ✓ 그리퍼를 열고 닫는 것을 학습
- ✓ 큐브를 잡는 것을 학습
- ✓ 큐브를 들어올리면 +50 보상

### Step 3: 학습된 모델 평가
```bash
# 가장 최근 학습 디렉토리 찾기
ls -lt models/ | head -5

# 평가 실행 (예시)
python3 scripts/evaluate_robotiq.py \
    --model-path models/ppo_ur5e_robotiq_pick_easy_YYYYMMDD_HHMMSS/best/best_model.zip \
    --n-episodes 10 \
    --slow-motion
```

## 3. Full Task 학습 (고급)

### 정상 난이도로 학습 (2-3시간)
```bash
python3 scripts/train_ppo_robotiq.py \
    --task-mode pick_place \
    --total-timesteps 2000000 \
    --n-envs 16 \
    --learning-rate 3e-4
```

### TensorBoard로 모니터링
```bash
# 새 터미널에서
tensorboard --logdir logs/
# http://localhost:6006 에서 확인
```

## 4. 학습 파라미터 조정

### 탐색 증가 (학습이 정체되었을 때)
```bash
python3 scripts/train_ppo_robotiq.py \
    --ent-coef 0.05 \
    --learning-rate 1e-4
```

### 더 빠른 학습 (많은 CPU 있을 때)
```bash
python3 scripts/train_ppo_robotiq.py \
    --n-envs 16 \
    --batch-size 512
```

### GPU 사용
```bash
python3 scripts/train_ppo_robotiq.py \
    --device cuda \
    --n-envs 16
```

## 5. 일반적인 문제 해결

### 문제: 그리퍼가 큐브를 못 잡음
**해결책:**
```bash
# Easy mode로 시작
python3 scripts/train_ppo_robotiq.py --easy-mode --task-mode pick

# 또는 XML 파일에서 마찰력 증가
# assets/ur5e_robotiq_pick_place.xml 의 friction="1.5" -> "2.0"
```

### 문제: 학습이 너무 느림
**해결책:**
```bash
# 병렬 환경 증가
python3 scripts/train_ppo_robotiq.py --n-envs 16

# 또는 프레임 스킵 증가 (환경 파일에서 수정)
```

### 문제: 보상이 -100에서 정체
**해결책:**
```bash
# 엔트로피 계수 증가
python3 scripts/train_ppo_robotiq.py --ent-coef 0.05

# Reach task부터 다시 시작
python3 scripts/train_ppo_robotiq.py --task-mode reach --easy-mode
```

## 6. 학습 결과 확인

### 성공률 확인
```bash
python3 scripts/evaluate_robotiq.py \
    --model-path models/<run_name>/best/best_model.zip \
    --n-episodes 50 \
    --deterministic
```

### 정성적 평가
```bash
# 슬로우 모션으로 동작 관찰
python3 scripts/evaluate_robotiq.py \
    --model-path models/<run_name>/best/best_model.zip \
    --slow-motion \
    --n-episodes 5
```

## 7. 다음 단계

### 커리큘럼 학습
1. Reach (easy mode) → 50만 스텝
2. Pick (easy mode) → 100만 스텝
3. Pick-Place (easy mode) → 200만 스텝
4. Pick-Place (normal mode) → 300만 스텝

### 파라미터 튜닝
- `learning_rate`: 1e-4 ~ 1e-3 시도
- `ent_coef`: 0.01 ~ 0.05 시도
- `n_envs`: 8 ~ 32 시도 (CPU에 따라)

### 고급 기능
- 목표 위치 랜덤화: `--randomize-target`
- Sparse reward: `--reward-type sparse`
- 커스텀 보상 함수 작성

## 유용한 명령어 모음

```bash
# 가상환경 활성화
source /home/yeseul/Desktop/XLeRobot/simulation/Maniskill/.manyvenv/bin/activate

# 환경 테스트
python3 test_robotiq_env.py

# 랜덤 데모
python3 scripts/evaluate_robotiq.py --random --n-episodes 3

# Easy mode 학습
python3 scripts/train_ppo_robotiq.py --easy-mode --task-mode pick --total-timesteps 1000000

# 모델 평가
python3 scripts/evaluate_robotiq.py --model-path <PATH> --n-episodes 20

# TensorBoard
tensorboard --logdir logs/

# 학습 중단 후 재개
python3 scripts/train_ppo_robotiq.py --model-path models/<run_name>/final_model.zip
```

## 성능 목표

### Reach Task
- **목표**: 성공률 > 80%
- **예상 학습 시간**: 30분 (50만 스텝)
- **지표**: `distance_to_cube < 0.05`

### Pick Task
- **목표**: 성공률 > 60%
- **예상 학습 시간**: 1시간 (100만 스텝)
- **지표**: `cube_grasped = True`, `cube_height > 0.45`

### Pick-Place Task
- **목표**: 성공률 > 40%
- **예상 학습 시간**: 2-3시간 (200만 스텝)
- **지표**: `distance_to_target < 0.05`, `cube_height > 0.42`

## 추가 리소스

- 전체 문서: `README_ROBOTIQ.md`
- 환경 코드: `envs/ur5e_robotiq_pick_place_env.py`
- XML 파일: `assets/ur5e_robotiq_pick_place.xml`
