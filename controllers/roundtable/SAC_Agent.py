# sac_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# 디바이스 설정 (CUDA 사용 가능 시 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================================================
# ReplayBuffer: 경험을 저장하고 샘플링하는 간단한 원형 버퍼 클래스
# ==================================================================================
class ReplayBuffer:
    """
    ReplayBuffer 클래스: 경험(transition)들을 저장하고 랜덤 샘플링을 제공합니다.

    :param max_size: 버퍼에 저장할 최대 transition 개수
    :param state_dim: 상태(state)의 차원
    :param action_dim: 행동(action)의 차원
    """
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # NumPy 배열로 버퍼 초기화
        self.state_buffer = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state_buffer = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros((max_size, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        """
        버퍼에 transition 저장

        :param state: 현재 상태(state)
        :param action: 취한 행동(action)
        :param reward: 보상(reward)
        :param next_state: 다음 상태(next_state)
        :param done: 에피소드 종료 여부(done flag)
        """
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = next_state
        self.done_buffer[self.ptr] = done

        # 포인터 이동 및 크기 갱신
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        """
        랜덤하게 배치 크기만큼 transition을 샘플링하여 반환

        :param batch_size: 샘플링할 배치 크기
        :return: (states, actions, rewards, next_states, dones) 튜플 (torch.Tensor)
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        states = torch.tensor(self.state_buffer[idxs]).to(device)
        actions = torch.tensor(self.action_buffer[idxs]).to(device)
        rewards = torch.tensor(self.reward_buffer[idxs]).to(device)
        next_states = torch.tensor(self.next_state_buffer[idxs]).to(device)
        dones = torch.tensor(self.done_buffer[idxs]).to(device)
        return states, actions, rewards, next_states, dones


# ==================================================================================
# CriticNetwork: Q-value를 근사하는 Critic 네트워크 (두 개를 병렬로 운용)
# ==================================================================================
class CriticNetwork(nn.Module):
    """
    CriticNetwork 클래스: 상태-행동 쌍 (s,a)에 대한 Q 값을 근사합니다.

    :param state_dim: 상태(state)의 차원
    :param action_dim: 행동(action)의 차원
    :param hidden_sizes: 은닉층 노드 크기 튜플, 예: (256, 256)
    """
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(CriticNetwork, self).__init__()
        input_dim = state_dim + action_dim
        # 첫 번째 은닉층
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        # 두 번째 은닉층
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        # 최종 출력층 (Q-value 스칼라)
        self.q_out = nn.Linear(hidden_sizes[1], 1)

        # 가중치 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.q_out.weight)
        nn.init.constant_(self.q_out.bias, 0.0)

    def forward(self, state, action):
        """
        순전파: Q(s, a) 계산

        :param state: 상태 텐서 (batch, state_dim)
        :param action: 행동 텐서 (batch, action_dim)
        :return: Q-value 텐서 (batch, 1)
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value


# ==================================================================================
# PolicyNetwork: 정책(action)을 샘플링하거나 결정론적(mean)을 반환하는 네트워크
# ==================================================================================
class PolicyNetwork(nn.Module):
    """
    PolicyNetwork 클래스: 상태(state)를 입력받아 행동(action)을 샘플링하거나 결정론적으로 반환합니다.

    :param state_dim: 상태(state)의 차원
    :param action_dim: 행동(action)의 차원 (여기서는 2: roll, pitch)
    :param hidden_sizes: 은닉층 노드 크기 튜플, 예: (256, 256)
    :param action_range: 행동 범위 튜플 (최솟값, 최댓값), 예: (-1.57, +1.57)
    :param log_std_min: log_std 분포 최소값
    :param log_std_max: log_std 분포 최대값
    """
    def __init__(self, state_dim, action_dim, hidden_sizes, action_range, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_low, self.action_high = action_range

        # 상태 → 은닉층
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # 평균(mean) 및 표준편차 로그(log_std) 출력층
        self.mean_layer = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[1], action_dim)

        # 가중치 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        nn.init.xavier_uniform_(self.log_std_layer.weight)
        nn.init.constant_(self.log_std_layer.bias, 0.0)

    def forward(self, state):
        """
        순전파: 상태(state) → 은닉층 → 평균(mean)과 log_std →
        Normal 분포로부터 재매개변수화 샘플링 후 tanh 및 스케일 적용

        :param state: 상태 텐서 (batch, state_dim)
        :return:
            action: 스케일된 행동 텐서 (batch, action_dim)
            log_prob: 해당 행동의 log 확률 (batch, 1)
            mean: 결정론적 행동(평균)을 스케일 적용한 텐서 (batch, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # 정규분포로부터 샘플링 (재매개변수화)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # 재매개변수화 객체
        tanh_z = torch.tanh(z)

        # 행동을 원래 범위로 스케일
        action = tanh_z * ((self.action_high - self.action_low) / 2.0) + \
                 ((self.action_high + self.action_low) / 2.0)

        # log_prob 계산 시 tanh 역함수 보정 (변수 변환 공식)
        log_prob = normal.log_prob(z) - torch.log(1 - tanh_z.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # 결정론적 행동 (평균)도 같은 스케일 적용
        mean_tanh = torch.tanh(mean)
        mean_action = mean_tanh * ((self.action_high - self.action_low) / 2.0) + \
                      ((self.action_high + self.action_low) / 2.0)

        return action, log_prob, mean_action


# ==================================================================================
# SACAgent: Soft Actor-Critic 알고리즘을 구현한 에이전트 클래스
# ==================================================================================
class SACAgent:
    """
    SACAgent 클래스: Soft Actor-Critic 알고리즘을 통해 연속 행동을 학습합니다.

    :param state_dim: 상태(state)의 차원
    :param action_dim: 행동(action)의 차원
    :param hidden_sizes: 은닉층 크기 튜플, 예: (256,256)
    :param gamma: 할인율 (discount factor)
    :param lr: 학습률 (Adam optimizer용)
    :param replay_size: 리플레이 버퍼 최대 크기
    :param batch_size: 배치 크기
    :param tau: 타겟 네트워크 하드업데이트 비율 (soft update)
    :param alpha: 엔트로피 계수 (고정값)
    """
    def __init__(self, state_dim, action_dim, hidden_sizes,
                 gamma=0.99, lr=3e-4, replay_size=100000, batch_size=256,
                 tau=0.005, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha

        # 행동 범위: roll, pitch는 [-1.57, +1.57]
        self.action_range = (-1.57, 1.57)

        # 리플레이 버퍼 초기화
        self.replay_buffer = ReplayBuffer(max_size=replay_size,
                                          state_dim=state_dim,
                                          action_dim=action_dim)

        # 네트워크 초기화
        # Critic 네트워크 1, 2
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_sizes).to(device)
        # 타겟 Critic 네트워크 1, 2 (하드 복제 후 매 업데이트마다 소프트 업데이트)
        self.critic1_target = CriticNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.critic2_target = CriticNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)

        # Policy 네트워크 (actor)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_sizes,
                                    action_range=self.action_range).to(device)

        # 옵티마이저
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def _hard_update(self, target_net, source_net):
        """
        하드 업데이트: source 네트워크의 파라미터를 target 네트워크에 그대로 복사

        :param target_net: 대상 네트워크 (torch.nn.Module)
        :param source_net: 소스 네트워크 (torch.nn.Module)
        """
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(param.data)

    def _soft_update(self, target_net, source_net):
        """
        소프트 업데이트: target = tau * source + (1‐tau) * target

        :param target_net: 대상 네트워크 (torch.nn.Module)
        :param source_net: 소스 네트워크 (torch.nn.Module)
        """
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state, evaluate=False):
        """
        행동(action)을 선택하여 반환.
        :param state: 상태 배열 또는 리스트 (길이 state_dim)
        :param evaluate: True면 deterministic(평균) 행동, False면 샘플링 행동
        :return: 행동 배열 (action_dim)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            if evaluate:
                # 결정론적 행동 (평균) 반환
                _, _, mean_action = self.policy(state_tensor)
                action = mean_action.cpu().numpy().flatten()
            else:
                # 샘플링된 행동 반환
                action, _, _ = self.policy(state_tensor)
                action = action.cpu().numpy().flatten()

        # 반환값이 NumPy 배열 형태 (roll, pitch)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        리플레이 버퍼에 경험(transition) 저장
        :param state: 상태 (array-like)
        :param action: 행동 (array-like)
        :param reward: 보상 (스칼라 또는 shape=(1,))
        :param next_state: 다음 상태 (array-like)
        :param done: 에피소드 종료 여부 (0.0 또는 1.0)
        """
        self.replay_buffer.store(state, action, reward, next_state, done)

    def train_step(self):
        """
        한 번의 학습 업데이트를 수행합니다.
        1) 리플레이 버퍼에서 배치를 샘플링
        2) Critic loss 계산 및 업데이트 (두 개의 Q 네트워크)
        3) Policy loss 계산 및 업데이트
        4) Target Critic 네트워크 소프트 업데이트
        """
        if self.replay_buffer.size < self.batch_size:
            return

        # 배치 샘플링
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample_batch(self.batch_size)

        ### 1) Critic 업데이트 ###
        # 다음 상태에서 policy로부터 행동 샘플 및 log_prob 계산
        next_actions, next_log_probs, _ = self.policy(next_states)
        # 타겟 Critic으로부터 Q 값 계산
        target_q1 = self.critic1_target(next_states, next_actions)
        target_q2 = self.critic2_target(next_states, next_actions)
        # 두 개 중 작은 Q 값 사용
        target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
        # 벨만 타깃: r + γ * (1‐done) * target_q
        target_value = rewards + (1 - dones) * self.gamma * target_q

        # 현재 Critic 값
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # Critic loss = MSE(current_q, target_value.detach())
        critic1_loss = F.mse_loss(current_q1, target_value.detach())
        critic2_loss = F.mse_loss(current_q2, target_value.detach())

        # Critic1 업데이트
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Critic2 업데이트
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        ### 2) Policy 업데이트 ###
        # 현재 상태에서 정책 네트워크로부터 행동 샘플 및 log_prob 계산
        new_actions, log_probs, _ = self.policy(states)
        # Q 값 추정
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        # 두 개 중 작은 Q 값 사용
        q_new_min = torch.min(q1_new, q2_new)
        # 정책 손실 = (α * log_probs - Q) 평균
        policy_loss = (self.alpha * log_probs - q_new_min).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        ### 3) Target Critic 네트워크 소프트 업데이트 ###
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

    def save(self, file_path):
        """
        에이전트의 네트워크 파라미터를 저장
        :param file_path: 저장할 .pth 파일 경로
        """
        torch.save({
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict()
        }, file_path)

    def load(self, file_path):
        """
        저장된 .pth 파일로부터 네트워크 파라미터를 불러옵니다.
        :param file_path: 불러올 .pth 파일 경로
        """
        checkpoint = torch.load(file_path, map_location=device)
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])

        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
