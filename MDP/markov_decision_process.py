import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from enum import Enum

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class GridWorldMDP:
    """
    Grid World 환경을 사용한 Markov Decision Process 구현
    """
    
    def __init__(self, height: int = 4, width: int = 4, 
                 terminal_states: List[Tuple[int, int]] = None,
                 rewards: Dict[Tuple[int, int], float] = None,
                 gamma: float = 0.9):
        """
        Args:
            height: 그리드 높이
            width: 그리드 너비
            terminal_states: 종료 상태들
            rewards: 각 상태별 보상
            gamma: 할인 인자
        """
        self.height = height
        self.width = width
        self.gamma = gamma
        
        # 기본 종료 상태 설정 (좌상단, 우하단)
        if terminal_states is None:
            self.terminal_states = [(0, 0), (height-1, width-1)]
        else:
            self.terminal_states = terminal_states
            
        # 기본 보상 설정
        if rewards is None:
            self.rewards = {(0, 0): 1.0, (height-1, width-1): -1.0}
        else:
            self.rewards = rewards
            
        # 상태 공간
        self.states = [(i, j) for i in range(height) for j in range(width)]
        self.num_states = len(self.states)
        
        # 행동 공간
        self.actions = list(Action)
        self.num_actions = len(self.actions)
        
        # 전이 확률 행렬 초기화
        self.transition_probs = self._build_transition_matrix()
        
    def _build_transition_matrix(self) -> np.ndarray:
        """전이 확률 행렬 구축"""
        # P[s][a][s'] = P(s'|s,a)
        P = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        for s_idx, state in enumerate(self.states):
            if state in self.terminal_states:
                # 종료 상태에서는 자기 자신으로만 전이
                P[s_idx, :, s_idx] = 1.0
            else:
                for a_idx, action in enumerate(self.actions):
                    next_state = self._get_next_state(state, action)
                    next_s_idx = self.states.index(next_state)
                    P[s_idx, a_idx, next_s_idx] = 1.0
                    
        return P
    
    def _get_next_state(self, state: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """주어진 상태와 행동에 대한 다음 상태 반환"""
        i, j = state
        
        if action == Action.UP:
            next_i = max(0, i - 1)
            next_j = j
        elif action == Action.DOWN:
            next_i = min(self.height - 1, i + 1)
            next_j = j
        elif action == Action.LEFT:
            next_i = i
            next_j = max(0, j - 1)
        elif action == Action.RIGHT:
            next_i = i
            next_j = min(self.width - 1, j + 1)
            
        return (next_i, next_j)
    
    def get_reward(self, state: Tuple[int, int]) -> float:
        """상태에 대한 보상 반환"""
        return self.rewards.get(state, 0.0)
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """종료 상태인지 확인"""
        return state in self.terminal_states
    
    def value_iteration(self, theta: float = 1e-6, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Value Iteration 알고리즘 구현
        
        Args:
            theta: 수렴 임계값
            max_iterations: 최대 반복 횟수
            
        Returns:
            V: 최적 가치 함수
            policy: 최적 정책
        """
        # 가치 함수 초기화
        V = np.zeros(self.num_states)
        
        for iteration in range(max_iterations):
            V_old = V.copy()
            
            for s_idx, state in enumerate(self.states):
                if self.is_terminal(state):
                    continue
                    
                # 모든 행동에 대한 Q값 계산
                q_values = []
                for a_idx in range(self.num_actions):
                    q_value = 0
                    for next_s_idx in range(self.num_states):
                        next_state = self.states[next_s_idx]
                        prob = self.transition_probs[s_idx, a_idx, next_s_idx]
                        reward = self.get_reward(next_state)
                        q_value += prob * (reward + self.gamma * V_old[next_s_idx])
                    q_values.append(q_value)
                
                # 최대 Q값으로 가치 함수 업데이트
                V[s_idx] = max(q_values)
            
            # 수렴 확인
            if np.max(np.abs(V - V_old)) < theta:
                print(f"Value Iteration 수렴: {iteration + 1} 반복")
                break
        
        # 최적 정책 추출
        policy = self._extract_policy(V)
        
        return V, policy
    
    def policy_iteration(self, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Policy Iteration 알고리즘 구현
        
        Args:
            max_iterations: 최대 반복 횟수
            
        Returns:
            V: 최적 가치 함수
            policy: 최적 정책
        """
        # 랜덤 정책으로 초기화
        policy = np.random.randint(0, self.num_actions, self.num_states)
        
        for iteration in range(max_iterations):
            # Policy Evaluation
            V = self._policy_evaluation(policy)
            
            # Policy Improvement
            new_policy = self._extract_policy(V)
            
            # 정책이 변하지 않으면 수렴
            if np.array_equal(policy, new_policy):
                print(f"Policy Iteration 수렴: {iteration + 1} 반복")
                break
                
            policy = new_policy
        
        return V, policy
    
    def _policy_evaluation(self, policy: np.ndarray, theta: float = 1e-6) -> np.ndarray:
        """주어진 정책에 대한 가치 함수 계산"""
        V = np.zeros(self.num_states)
        
        while True:
            V_old = V.copy()
            
            for s_idx, state in enumerate(self.states):
                if self.is_terminal(state):
                    continue
                    
                action = policy[s_idx]
                v = 0
                for next_s_idx in range(self.num_states):
                    next_state = self.states[next_s_idx]
                    prob = self.transition_probs[s_idx, action, next_s_idx]
                    reward = self.get_reward(next_state)
                    v += prob * (reward + self.gamma * V_old[next_s_idx])
                
                V[s_idx] = v
            
            if np.max(np.abs(V - V_old)) < theta:
                break
                
        return V
    
    def _extract_policy(self, V: np.ndarray) -> np.ndarray:
        """가치 함수로부터 정책 추출"""
        policy = np.zeros(self.num_states, dtype=int)
        
        for s_idx, state in enumerate(self.states):
            if self.is_terminal(state):
                continue
                
            q_values = []
            for a_idx in range(self.num_actions):
                q_value = 0
                for next_s_idx in range(self.num_states):
                    next_state = self.states[next_s_idx]
                    prob = self.transition_probs[s_idx, a_idx, next_s_idx]
                    reward = self.get_reward(next_state)
                    q_value += prob * (reward + self.gamma * V[next_s_idx])
                q_values.append(q_value)
            
            policy[s_idx] = np.argmax(q_values)
            
        return policy
    
    def visualize_policy(self, policy: np.ndarray, title: str = "Policy"):
        """정책 시각화"""
        policy_grid = np.zeros((self.height, self.width), dtype=object)
        
        action_symbols = {
            Action.UP: '↑',
            Action.DOWN: '↓', 
            Action.LEFT: '←',
            Action.RIGHT: '→'
        }
        
        for s_idx, state in enumerate(self.states):
            i, j = state
            if self.is_terminal(state):
                policy_grid[i, j] = 'T'
            else:
                action = Action(policy[s_idx])
                policy_grid[i, j] = action_symbols[action]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(np.ones((self.height, self.width)), cmap='lightgray')
        
        for i in range(self.height):
            for j in range(self.width):
                plt.text(j, i, policy_grid[i, j], ha='center', va='center', 
                        fontsize=20, fontweight='bold')
        
        plt.title(title, fontsize=16)
        plt.xticks(range(self.width))
        plt.yticks(range(self.height))
        plt.grid(True)
        plt.show()
    
    def visualize_values(self, V: np.ndarray, title: str = "Value Function"):
        """가치 함수 시각화"""
        value_grid = np.zeros((self.height, self.width))
        
        for s_idx, state in enumerate(self.states):
            i, j = state
            value_grid[i, j] = V[s_idx]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(value_grid, cmap='viridis')
        plt.colorbar()
        
        for i in range(self.height):
            for j in range(self.width):
                plt.text(j, i, f'{value_grid[i, j]:.2f}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        plt.title(title, fontsize=16)
        plt.xticks(range(self.width))
        plt.yticks(range(self.height))
        plt.show()


def main():
    """MDP 예제 실행"""
    print("=== Markov Decision Process 구현 예제 ===\n")
    
    # Grid World MDP 생성
    mdp = GridWorldMDP(height=4, width=4, gamma=0.9)
    
    print("환경 정보:")
    print(f"- 그리드 크기: {mdp.height} x {mdp.width}")
    print(f"- 종료 상태: {mdp.terminal_states}")
    print(f"- 보상: {mdp.rewards}")
    print(f"- 할인 인자: {mdp.gamma}\n")
    
    # Value Iteration
    print("1. Value Iteration 실행...")
    V_vi, policy_vi = mdp.value_iteration()
    print("Value Iteration 완료!\n")
    
    # Policy Iteration  
    print("2. Policy Iteration 실행...")
    V_pi, policy_pi = mdp.policy_iteration()
    print("Policy Iteration 완료!\n")
    
    # 결과 비교
    print("=== 결과 비교 ===")
    print("Value Iteration 가치 함수:")
    for i, state in enumerate(mdp.states):
        print(f"State {state}: {V_vi[i]:.4f}")
    
    print("\nPolicy Iteration 가치 함수:")
    for i, state in enumerate(mdp.states):
        print(f"State {state}: {V_pi[i]:.4f}")
    
    print(f"\n가치 함수 차이 (최대): {np.max(np.abs(V_vi - V_pi)):.6f}")
    print(f"정책 일치 여부: {np.array_equal(policy_vi, policy_pi)}")
    
    # 텍스트로 정책 출력
    print("\n=== 최적 정책 (텍스트) ===")
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    print("Value Iteration Policy:")
    for i in range(mdp.height):
        row = ""
        for j in range(mdp.width):
            state = (i, j)
            s_idx = mdp.states.index(state)
            if mdp.is_terminal(state):
                row += "T "
            else:
                row += action_symbols[policy_vi[s_idx]] + " "
        print(row)
    
    # 시각화 (matplotlib 사용 가능한 경우)
    try:
        print("\n그래픽 시각화 중...")
        mdp.visualize_policy(policy_vi, "Value Iteration Policy")
        mdp.visualize_values(V_vi, "Value Iteration Values")
    except:
        print("GUI 환경에서만 그래픽 시각화가 가능합니다.")


if __name__ == "__main__":
    main()