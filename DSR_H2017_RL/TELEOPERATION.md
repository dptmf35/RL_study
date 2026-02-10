# DSR H2017 Teleoperation Guide

MuJoCo 시뮬레이션 내에서 DSR H2017 로봇을 키보드로 직접 조작하여 최적의 home position을 찾을 수 있습니다.

## 🚀 실행 방법

```bash
cd DSR_H2017_RL
python3 scripts/teleop_dsr.py
```

### 선택사항: 실시간 키보드 제어

더 나은 사용자 경험을 위해 `keyboard` 라이브러리 설치:

```bash
pip install keyboard
```

**주의**: `keyboard` 라이브러리는 시스템 레벨 키보드 접근이 필요하므로 sudo 권한이 필요할 수 있습니다.

## 🎮 조작 방법

### 키보드 모드 (keyboard 라이브러리 설치 시)

실시간으로 키를 누르면 즉시 반응합니다.

#### Joint 제어
| 키 | 동작 |
|----|------|
| `1` / `Q` | Joint 1 (Base) 감소/증가 |
| `2` / `W` | Joint 2 (Shoulder) 감소/증가 |
| `3` / `E` | Joint 3 (Elbow) 감소/증가 |
| `4` / `R` | Joint 4 (Wrist1) 감소/증가 |
| `5` / `T` | Joint 5 (Wrist2) 감소/증가 |
| `6` / `Y` | Joint 6 (Wrist3) 감소/증가 |

#### Gripper 제어
| 키 | 동작 |
|----|------|
| `G` | Gripper 닫기 |
| `H` | Gripper 열기 |

#### Step Size 조정
| 키 | 동작 |
|----|------|
| `-` | Step size 감소 (더 세밀한 제어) |
| `=` | Step size 증가 (더 빠른 이동) |

#### 기타 기능
| 키 | 동작 |
|----|------|
| `SPACE` | 저장된 home position으로 리셋 |
| `P` | 현재 joint 각도를 코드로 출력 (복사하여 사용) |
| `S` | 현재 위치를 새로운 home으로 저장 |
| `I` | 상태 정보 출력 (위치, 거리, 방향) |
| `ESC` / `C` | 종료 |

### 수동 입력 모드 (keyboard 라이브러리 없이)

명령어를 타이핑하고 Enter를 눌러 제어합니다.

#### 명령어
- `j1+` / `j1-` : Joint 1 증가/감소
- `j2+` / `j2-` : Joint 2 증가/감소
- `j3+` / `j3-` : Joint 3 증가/감소
- `j4+` / `j4-` : Joint 4 증가/감소
- `j5+` / `j5-` : Joint 5 증가/감소
- `j6+` / `j6-` : Joint 6 증가/감소
- `g+` / `g-` : Gripper 열기/닫기
- `home` : Home position으로 리셋
- `print` / `p` : 현재 각도를 코드로 출력
- `save` / `s` : 현재 위치 저장
- `info` / `i` : 상태 정보 출력
- `quit` / `q` : 종료
- `help` / `h` : 도움말 표시

## 📋 사용 워크플로우

### 1. Home Position 찾기

```bash
python3 scripts/teleop_dsr.py
```

1. MuJoCo viewer가 열리면서 로봇이 현재 home position에 위치
2. 키보드로 각 joint를 조작하여 원하는 자세 설정
3. `I` 키를 눌러 상태 정보 확인:
   - End-effector 위치
   - Cube까지 거리
   - Gripper 방향 (downward alignment)

### 2. 최적 위치 저장

만족스러운 자세를 찾았으면:

1. `S` 키로 현재 위치를 home으로 저장
2. `P` 키로 코드 출력
3. 출력된 코드를 복사

### 3. 코드에 적용

출력된 코드를 `envs/dsr_h2017_align_env.py` 파일의 `home_qpos`에 붙여넣기:

```python
# Line ~114
self.home_qpos = np.array([
     0.0000,  # Base
    -0.1000,  # Shoulder
     1.8000,  # Elbow
    -2.5000,  # Wrist1
    -1.4000,  # Wrist2
     0.0000,  # Wrist3
])
```

### 4. 검증

```bash
python3 scripts/debug_orientation.py
python3 scripts/check_arm_horizontal.py
python3 scripts/evaluate_align.py --random --n-episodes 5
```

## 💡 팁

### 좋은 Home Position 찾기

1. **목표**:
   - End-effector가 cube spawn 영역(X: 0.6-0.8, Z: ~0.88) 근처
   - Gripper가 아래를 향함 (downward alignment > 0.7)
   - 초기 거리가 적당히 가까움 (0.1-0.4m)

2. **전략**:
   - Joint 2 (Shoulder): 팔 높이 조절 (-0.3 ~ 0.2 범위)
   - Joint 3 (Elbow): 전방 도달 거리 (1.5 ~ 2.5 범위)
   - Joint 4, 5 (Wrist): Gripper 방향 (-2.5 ~ -1.0 범위)

3. **확인 사항**:
   - `I` 키로 downward alignment가 0.7 이상인지 확인
   - Distance XY가 0.1-0.5m 범위인지 확인
   - End-effector Z가 0.85-0.90m 범위인지 확인

### Step Size 조정

- **거친 조정**: Step size 0.1-0.2 (큰 이동)
- **미세 조정**: Step size 0.01-0.05 (정밀 제어)

### 실험하기

- `SPACE`로 언제든 home으로 돌아갈 수 있으니 자유롭게 실험
- `S`로 여러 후보를 저장하면서 비교 가능

## 🔧 문제 해결

### "keyboard library not found" 에러

```bash
pip install keyboard
```

설치 후에도 작동하지 않으면 수동 입력 모드로 사용하거나 sudo로 실행:

```bash
sudo python3 scripts/teleop_dsr.py
```

### MuJoCo viewer가 열리지 않음

MuJoCo와 OpenGL이 제대로 설치되었는지 확인:

```bash
python3 -c "import mujoco; print(mujoco.__version__)"
```

### Joint가 움직이지 않음

Joint limit에 도달했을 수 있습니다. 반대 방향으로 조작하거나 다른 joint를 사용하세요.

## 📊 출력 정보 이해하기

### Joint Angles 출력 예시

```
Joint angles:
  Base     (J1):  0.0000 rad (   0.0°)
  Shoulder (J2): -0.1000 rad (  -5.7°)
  Elbow    (J3):  1.8000 rad ( 103.1°)
  Wrist1   (J4): -2.5000 rad (-143.2°)
  Wrist2   (J5): -1.4000 rad ( -80.2°)
  Wrist3   (J6):  0.0000 rad (   0.0°)
  Gripper:  0.000
```

### State Info 출력 예시

```
CURRENT STATE INFO
End-effector: (0.662, 0.173, 0.867)  ← Gripper 위치
Cube:         (0.724, -0.056, 0.780)  ← Cube 위치
Distance XY:  0.088m                  ← 수평 거리 (가까울수록 좋음)
Distance 3D:  0.132m                  ← 전체 거리
Gripper Z-axis: (0.067, 0.590, -0.805) ← 방향 벡터
Downward alignment: 0.805             ← 수직도 (1.0 = 완벽)
```

## 🎯 추천 설정 찾기 프로세스

1. **시작**: 현재 home position에서 시작
2. **Shoulder 조정**: Joint 2로 전체 팔 높이 조절
3. **Elbow 조정**: Joint 3으로 전방 reach 조절
4. **Wrist 미세조정**: Joint 4, 5로 gripper 방향 조절
5. **검증**: `I`로 상태 확인, downward > 0.7, distance < 0.5m 목표
6. **저장**: 만족스러우면 `S`와 `P`로 저장 및 코드 복사
7. **재시도**: `SPACE`로 리셋 후 다른 설정 시도
8. **최종 선택**: 여러 후보 중 최선 선택

Happy teleoperating! 🤖
