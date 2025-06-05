import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

############################
# [1] RQNAgent (DQN 계열)
############################
class RQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q값 반환


class RQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim  # 이산 행동 개수 (예: 27)
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 1e-3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RQNetwork(state_dim, action_dim).to(self.device)
        self.target_model = RQNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 타깃 네트워크 초기동기화
        self.update_target_network()

        # 행동 후보 맵핑 (이산→연속 속도)
        # 예: 속도 후보 = [-1.0, 0.0, +1.0] 
        # 조합으로 총 27개 행동 매핑
        self.vel_candidates = [-1.0, 0.0, 1.0]
        self.discrete_to_vel = []
        for v1 in self.vel_candidates:
            for v2 in self.vel_candidates:
                for v3 in self.vel_candidates:
                    self.discrete_to_vel.append([v1, v2, v3])

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def map_discrete_to_velocity(self, action_idx):
        return self.discrete_to_vel[action_idx]

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def store_transition(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s,a) 예측
        q_pred = self.model(states).gather(1, actions)

        # 타깃 Q값 계산: y = r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_model(next_states)
            max_q_next, _ = torch.max(q_next, dim=1, keepdim=True)
            q_target = rewards + (1 - dones) * self.gamma * max_q_next

        # 손실 및 업데이트
        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε 스케줄링
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 타깃 네트워크 주기 업데이트
        # 예: 일정 스텝마다 또는 소프트 업데이트
        # 여기서는 간단히 하드 업데이트 예시 (매 학습마다)
        self.update_target_network()


############################
# [2] SACAgent
############################
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)  # 안정적인 log_std 범위
        std = torch.exp(log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q(x)


class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 3e-4
        self.batch_size = 64

        # 장치 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 네트워크 초기화
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim).to(self.device)

        # 타깃 네트워크 동기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)

        # 온도 파라미터 α
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        self.target_entropy = -action_dim

        # Replay Buffer
        self.memory = deque(maxlen=100000)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, std = self.actor(state)
        if evaluate:
            action = torch.tanh(mu)
            return action.cpu().detach().numpy()[0]
        else:
            dist = torch.distributions.Normal(mu, std)
            z = dist.rsample()  # reparameterization trick
            action = torch.tanh(z)
            return action.cpu().detach().numpy()[0]

    def store_transition(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # 미니배치 샘플링
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.FloatTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 1) Critic 업데이트
        with torch.no_grad():
            mu_next, std_next = self.actor(next_states)
            dist_next = torch.distributions.Normal(mu_next, std_next)
            z_next = dist_next.rsample()
            a_next = torch.tanh(z_next)
            log_prob_next = dist_next.log_prob(z_next).sum(dim=1, keepdim=True) - torch.log(1 - a_next.pow(2) + 1e-6).sum(dim=1, keepdim=True)

            q1_next = self.target_critic1(next_states, a_next)
            q2_next = self.target_critic2(next_states, a_next)
            q_next = torch.min(q1_next, q2_next) - torch.exp(self.log_alpha) * log_prob_next
            q_target = rewards + (1 - dones) * self.gamma * q_next

        # 현재 Critic 값
        q1_pred = self.critic1(states, actions)
        q2_pred = self.critic2(states, actions)
        critic1_loss = nn.MSELoss()(q1_pred, q_target)
        critic2_loss = nn.MSELoss()(q2_pred, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 2) Actor 업데이트
        mu, std = self.actor(states)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a_t = torch.tanh(z)
        log_prob = dist.log_prob(z).sum(dim=1, keepdim=True) - torch.log(1 - a_t.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        q1_val = self.critic1(states, a_t)
        q2_val = self.critic2(states, a_t)
        q_val = torch.min(q1_val, q2_val)

        actor_loss = (torch.exp(self.log_alpha) * log_prob - q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3) α 업데이트 (온도 파라미터)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 4) 타깃 네트워크 업데이트 (소프트 업데이트)
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self):
        self.update()
