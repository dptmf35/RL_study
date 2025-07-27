# 로봇 강화학습 입문 - 2단계: PyBullet 로봇 환경
# 필요한 설치: pip install pybullet gymnasium stable-baselines3

import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
import time
import os

class SimpleRobotEnv(gym.Env):
    """간단한 PyBullet 로봇 환경 직접 구현"""
    
    def __init__(self, render_mode=None):
        super(SimpleRobotEnv, self).__init__()
        
        # 행동 공간: 2개 바퀴의 속도 (-1 ~ 1)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        
        # 관측 공간: [x, y, orientation, x_vel, y_vel, ang_vel]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.physics_client = None
        self.robot_id = None
        self.target_pos = [2, 2, 0]
        self.max_steps = 1000
        self.current_step = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # PyBullet 초기화
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # 바닥 생성
        p.loadURDF("plane.urdf")
        
        # 간단한 로봇 생성 (상자 + 바퀴)
        self.robot_id = self._create_simple_robot()
        
        # 목표점 시각화
        p.loadURDF("sphere2.urdf", self.target_pos, globalScaling=0.1)
        
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def _create_simple_robot(self):
        """간단한 2바퀴 로봇 생성"""
        # 몸체
        body_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.05])
        body_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.05], rgbaColor=[0, 0, 1, 1])
        
        # 바퀴
        wheel_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.02)
        wheel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.02, rgbaColor=[1, 0, 0, 1])
        
        # 로봇 조립
        robot_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=[0, 0, 0.1],
            linkMasses=[0.1, 0.1],
            linkCollisionShapeIndices=[wheel_collision, wheel_collision],
            linkVisualShapeIndices=[wheel_visual, wheel_visual],
            linkPositions=[[0.1, -0.15, 0], [0.1, 0.15, 0]],
            linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 1, 0], [0, 1, 0]]
        )
        
        return robot_id
    
    def step(self, action):
        # 바퀴 속도 적용
        left_wheel_vel = action[0] * 10  # 최대 속도 10 rad/s
        right_wheel_vel = action[1] * 10
        
        p.setJointMotorControl2(
            self.robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=left_wheel_vel
        )
        p.setJointMotorControl2(
            self.robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=right_wheel_vel
        )
        
        # 물리 시뮬레이션 실행
        p.stepSimulation()
        
        # 관측값 및 보상 계산
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        self.current_step += 1
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        """로봇의 현재 상태 관측"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # 오일러 각도로 변환
        euler = p.getEulerFromQuaternion(orn)
        
        obs = np.array([
            pos[0], pos[1],  # x, y 위치
            euler[2],        # z축 회전각
            vel[0], vel[1],  # x, y 속도
            ang_vel[2]       # 각속도
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self):
        """보상 함수"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # 목표점까지의 거리
        distance = np.sqrt((pos[0] - self.target_pos[0])**2 + 
                          (pos[1] - self.target_pos[1])**2)
        
        # 거리에 반비례하는 보상
        reward = -distance
        
        # 목표점 근처 도착 시 큰 보상
        if distance < 0.2:
            reward += 100
            
        # 맵 밖으로 나가면 패널티
        if abs(pos[0]) > 5 or abs(pos[1]) > 5:
            reward -= 50
            
        return reward
    
    def _is_terminated(self):
        """종료 조건"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance = np.sqrt((pos[0] - self.target_pos[0])**2 + 
                          (pos[1] - self.target_pos[1])**2)
        
        # 목표점 도착 또는 맵 밖으로 나감
        return distance < 0.2 or abs(pos[0]) > 5 or abs(pos[1]) > 5
    
    def render(self):
        if self.render_mode == "human":
            time.sleep(0.01)  # 시각화 속도 조절
    
    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

def train_simple_robot():
    """간단한 로봇 학습"""
    print("=== PyBullet 간단한 로봇 학습 ===")
    
    # 환경 생성
    env = SimpleRobotEnv(render_mode=None)
    
    print(f"관측 공간: {env.observation_space}")
    print(f"행동 공간: {env.action_space}")
    
    # PPO 모델 생성
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003)
    
    # 학습
    print("학습 시작...")
    model.learn(total_timesteps=50000)
    
    # 모델 저장
    model.save("simple_robot_ppo")
    
    return model, env

def test_trained_robot():
    """학습된 로봇 테스트"""
    print("\n=== 학습된 로봇 테스트 ===")
    
    # 환경 생성 (시각화 모드)
    env = SimpleRobotEnv(render_mode="human")
    
    # 학습된 모델 로드
    try:
        model = PPO.load("simple_robot_ppo")
        print("저장된 모델을 로드했습니다.")
    except:
        print("저장된 모델이 없습니다. 먼저 학습을 실행하세요.")
        return
    
    # 테스트 실행
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        
        if terminated or truncated:
            print(f"에피소드 종료 - 총 보상: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
    
    env.close()

def try_builtin_envs():
    """PyBullet 내장 환경들 시도"""
    print("\n=== PyBullet 내장 환경 테스트 ===")
    
    # 사용 가능한 환경들 시도
    test_envs = [
        'CartPole-v1',  # 기본 환경
        'Pendulum-v1',  # 연속 제어 환경
    ]
    
    for env_name in test_envs:
        try:
            print(f"\n{env_name} 테스트 중...")
            env = gym.make(env_name, render_mode='rgb_array')
            
            # 간단한 랜덤 정책 테스트
            obs, info = env.reset()
            total_reward = 0
            
            for _ in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            print(f"{env_name} - 랜덤 정책 보상: {total_reward:.2f}")
            env.close()
            
        except Exception as e:
            print(f"{env_name} 실행 실패: {e}")

def advanced_robot_training():
    """고급 로봇 학습 (SAC 알고리즘)"""
    print("\n=== 고급 로봇 학습 (SAC) ===")
    
    env = SimpleRobotEnv(render_mode="human")
    
    # SAC 모델 (연속 제어에 특화)
    model = SAC('MlpPolicy', env, verbose=1, learning_rate=0.0003)
    
    print("SAC 학습 시작...")
    model.learn(total_timesteps=30000)
    
    # 모델 저장
    model.save("simple_robot_sac")
    
    # 간단한 성능 평가
    obs, info = env.reset()
    total_rewards = []
    
    for episode in range(10):
        episode_reward = 0
        obs, info = env.reset()
        
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    print(f"SAC 평균 성능: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    env.close()

if __name__ == "__main__":
    print("🤖 PyBullet 로봇 강화학습을 시작합니다!")
    print("=" * 50)
    
    # 1. 내장 환경 테스트
    try_builtin_envs()
    
    # 2. 간단한 로봇 학습
    model, env = train_simple_robot()
    env.close()
    
    # 3. 학습된 로봇 테스트
    test_trained_robot()
    
    # 4. 고급 알고리즘 테스트
    advanced_robot_training()
    
    print("\n🎉 PyBullet 로봇 학습 완료!")
    print("다음 단계: 더 복잡한 환경이나 Isaac Lab으로 이동하세요.")