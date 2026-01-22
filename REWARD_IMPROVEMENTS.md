# UR5e Robotiq Pick Place Environment - Reward Function Improvements

## ManiSkill PickCube 패턴 적용 완료 ✅

### 1. Staged/Gated Reward 방식 (핵심 개선!)

**이전 방식:**
```python
# 모든 보상을 단순히 더함
reward = reach_reward + grasp_reward + place_reward
```

**개선 방식 (PickCube 스타일):**
```python
# Stage 1: Reaching (항상 활성)
reaching_reward = 1 - np.tanh(5 * dist_to_cube)

# Stage 2: Grasped (이진 보너스)
grasped_reward = 1.0 if is_grasped else 0.0

# Stage 3: Placing (is_grasped로 GATING!)
place_reward = (1 - np.tanh(5 * dist_to_target)) * float(is_grasped)

# Stage 4: Static (is_obj_placed로 GATING!)
static_reward = (1 - np.tanh(5 * qvel_norm)) * float(is_obj_placed)

# Success: 최대 보상으로 지배
if is_obj_placed and is_robot_static:
    success_reward = 5.0
```

**효과:**
- ❌ **꼼수 방지 1**: 로봇이 큐브를 안 잡고 목표 지점으로 가도 place_reward = 0 (is_grasped가 False이므로 masking됨)
- ❌ **꼼수 방지 2**: 로봇이 큐브를 목표에 "던지면" static_reward = 0 (is_obj_placed가 False이므로 masking됨)
- ✅ **올바른 행동 강화**: 잡고 → 이동 → 놓고 → 멈추는 순서로만 최대 보상

---

### 2. 물리 기반 잡기 판정 (Physics-based Grasp Detection)

**이전 방식:**
```python
# 거리 + 그리퍼 상태만 체크
if dist_to_cube < 0.08 and gripper_is_closed and cube_height > table_height + 0.02:
    self.cube_grasped = True  # 플래그 설정
```

**개선 방식:**
```python
def _check_grasp(self) -> bool:
    """
    엄격한 3-way 체크:
    1. 그리퍼가 닫혀있는가? (gripper_pos > 0.4)
    2. 큐브가 그리퍼에 가까운가? (< 6cm)
    3. 큐브가 테이블에서 들어올려졌는가? (> 2cm)
    
    세 조건이 모두 만족해야 grasped = True
    """
    return gripper_is_closed and dist_to_cube < 0.06 and cube_is_lifted
```

**효과:**
- ✅ **더 엄격한 판정**: 세 조건을 모두 만족해야만 잡은 것으로 인정
- ✅ **매 스텝 재평가**: 플래그가 아니라 매 스텝마다 물리 상태를 체크
- ✅ **미래 확장 가능**: Mujoco contact sensor를 추가하면 더 정확한 판정 가능

---

### 3. 정지 보상 추가 (Static Reward)

**추가된 기능:**
```python
# 로봇 관절 속도 체크
joint_vel = np.array([self.data.qvel[...] for jid in self.joint_ids])
is_robot_static = np.linalg.norm(joint_vel) < 0.2

# 정지 보상 (큐브가 목표에 있을 때만 활성)
static_reward = (1 - np.tanh(5 * qvel_norm)) * float(is_obj_placed)

# Success 조건에도 추가
if is_obj_placed and is_robot_static:
    success_reward = 5.0
```

**효과:**
- ❌ **"던지기" 방지**: 큐브를 목표에 던져서 순간적으로 위치만 맞추는 꼼수 차단
- ✅ **부드러운 배치**: 로봇이 큐브를 조심스럽게 놓고 멈추도록 유도
- ✅ **실제 로봇에 중요**: 실제 로봇에서는 안정적인 배치가 필수

---

### 4. tanh 기반 거리 보상 (Smooth Distance Reward)

**이전 방식:**
```python
# 구간별 계단식 보상
if dist < 0.30: reward = 0.5
elif dist < 0.15: reward = 1.0
elif dist < 0.08: reward = 1.5
```

**개선 방식:**
```python
# 부드러운 연속 함수
reward = 1 - np.tanh(5 * distance)

# 거리별 보상값:
# dist = 0.00 → reward = 1.00 (최대)
# dist = 0.10 → reward = 0.62
# dist = 0.20 → reward = 0.27
# dist = ∞   → reward = 0.00 (최소)
```

**효과:**
- ✅ **부드러운 gradient**: 신경망 학습에 더 좋은 gradient 제공
- ✅ **모든 거리에서 학습**: 멀리 있을 때도 학습 신호 제공
- ✅ **수렴 안정성**: 갑작스러운 보상 변화가 없어 학습이 안정적

---

## 코드 변경 사항

### 추가된 함수:
1. `_check_grasp()` - 물리 기반 잡기 판정
2. `_get_info()` 개선 - `is_grasped`, `is_obj_placed`, `is_robot_static` 추가

### 제거된 코드:
1. `self.cube_grasped` 플래그 제거 (물리 기반 체크로 대체)
2. `self.gripper_was_open_near_cube` 플래그 제거 (불필요)

### 변경된 함수:
1. `_compute_reward()` - 완전히 재작성 (PickCube 패턴)

---

## 학습 예상 효과

### 이전 문제점:
- 로봇이 큐브 근처만 가고 잡지 않음
- 큐브를 던져서 목표에 맞추려 시도
- 보상이 계단식이라 gradient가 불안정

### 개선 후 기대 효과:
- ✅ 단계별 학습 가능 (Reach → Grasp → Place → Static)
- ✅ 꼼수 원천 차단 (Gating으로 순서 강제)
- ✅ 부드러운 학습 커브 (tanh로 연속적 보상)
- ✅ 안정적인 성공률 (정지 조건 추가)

---

## 테스트 방법

```bash
# 환경 테스트
python -c "
from envs.ur5e_robotiq_pick_place_env import UR5eRobotiqPickPlaceEnv
env = UR5eRobotiqPickPlaceEnv(task_mode='pick_place')
obs, info = env.reset()
print('Initial info:', info)
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0:
        print(f'Step {i}: reward={reward:.3f}, info={info}')
    if terminated or truncated:
        break
"
```

---

## 참고 자료

- ManiSkill PickCube 환경: `/home/yeseul/Desktop/ManiSkill/mani_skill/envs/tasks/tabletop/pick_cube.py`
- 핵심 코드 라인: 162-192 (compute_dense_reward)
