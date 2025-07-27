# 로봇 강화학습 입문 - 1단계: 기본 환경
# 필요한 설치: pip install gymnasium stable-baselines3 matplotlib

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

def test_basic_rl():
    """기본 강화학습 환경으로 시작"""
    print("=== 1단계: 기본 CartPole 환경 테스트 ===")
    
    # 환경 생성
    env = gym.make('CartPole-v1', render_mode='human')
    
    print(f"관측 공간: {env.observation_space}")
    print(f"행동 공간: {env.action_space}")
    
    # PPO 모델 생성
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003)
    
    # 학습
    print("학습 시작...")
    model.learn(total_timesteps=20000)
    
    # 테스트
    print("학습된 모델 테스트...")
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"에피소드 종료 - 총 보상: {total_reward}")
            break
    
    env.close()
    return model

def test_pendulum():
    """연속 제어 환경 테스트"""
    print("\n=== 2단계: Pendulum 연속 제어 환경 ===")
    
    # 연속 제어 환경
    env = gym.make('Pendulum-v1', render_mode='human')
    
    print(f"관측 공간: {env.observation_space}")
    print(f"행동 공간: {env.action_space}")
    
    # PPO 모델 (연속 행동 공간용)
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.001)
    
    # 학습
    print("Pendulum 학습 시작...")
    model.learn(total_timesteps=30000)
    
    # 테스트
    print("학습된 모델 테스트...")
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"평균 보상: {total_reward/200:.2f}")
    env.close()
    return model

def plot_learning_curve():
    """학습 곡선 시각화"""
    print("\n=== 3단계: 학습 곡선 시각화 ===")
    
    # 벡터화된 환경 생성 (병렬 학습)
    env = make_vec_env('CartPole-v1', n_envs=4)
    
    # 평가용 환경
    eval_env = gym.make('CartPole-v1')
    
    # 콜백 설정 (학습 과정 기록)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/',
        log_path='./logs/', 
        eval_freq=2000,
        deterministic=True, 
        render=False
    )
    
    # 모델 생성
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/")
    
    # 학습 (콜백과 함께)
    model.learn(total_timesteps=50000, callback=eval_callback)
    
    print("학습 완료! tensorboard로 결과 확인 가능: tensorboard --logdir ./tensorboard/")
    
    env.close()
    eval_env.close()
    return model

def compare_algorithms():
    """다양한 알고리즘 비교"""
    print("\n=== 4단계: 알고리즘 비교 ===")
    
    from stable_baselines3 import A2C, DQN
    
    env = gym.make('CartPole-v1')
    algorithms = {
        'PPO': PPO('MlpPolicy', env, verbose=0),
        'A2C': A2C('MlpPolicy', env, verbose=0),
        'DQN': DQN('MlpPolicy', env, verbose=0)
    }
    
    results = {}
    
    for name, model in algorithms.items():
        print(f"{name} 학습 중...")
        model.learn(total_timesteps=20000)
        
        # 평가
        total_rewards = []
        for _ in range(10):
            obs, info = env.reset()
            episode_reward = 0
            
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
        
        results[name] = np.mean(total_rewards)
        print(f"{name} 평균 보상: {results[name]:.2f}")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    algorithms_names = list(results.keys())
    scores = list(results.values())
    
    plt.bar(algorithms_names, scores)
    plt.title('알고리즘별 성능 비교')
    plt.ylabel('평균 보상')
    plt.xlabel('알고리즘')
    plt.show()
    
    env.close()
    return results

if __name__ == "__main__":
    # 로그 디렉토리 생성
    os.makedirs('./logs', exist_ok=True)
    
    print("🤖 로봇 강화학습 입문 과정을 시작합니다!")
    print("=" * 50)
    
    # 1단계: 기본 환경
    model1 = test_basic_rl()
    
    # 2단계: 연속 제어
    model2 = test_pendulum()
    
    # 3단계: 학습 곡선
    model3 = plot_learning_curve()
    
    # 4단계: 알고리즘 비교
    results = compare_algorithms()
    
    print("\n🎉 입문 과정 완료!")
    print("다음 단계: PyBullet 환경으로 이동하세요.")