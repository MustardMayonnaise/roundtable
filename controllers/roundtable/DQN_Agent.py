import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# DQN을 위한 신경망 클래스
# 상태(state)를 입력받아 각 이산 행동에 대한 Q값을 출력하는 역할을 한다.
class DQNNetwork(nn.Module):
    # state_dim: 상태 차원 (예: 6)
    # action_dim: 행동 개수 (예: 49)
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        # 첫 번째 은닉층: 입력 차원 → 128 유닛
        self.fc1 = nn.Linear(state_dim, 128)
        # 두 번째 은닉층: 128 유닛 → 128 유닛
        self.fc2 = nn.Linear(128, 128)
        # 출력층: 128 유닛 → action_dim (각 행동에 대한 Q값)
        self.fc3 = nn.Linear(128, action_dim)

    # 순전파(forward) 함수: 상태 텐서를 받아서 Q값 벡터를 반환한다.
    # x: 상태 텐서 (batch_size × state_dim)
    def forward(self, x):
        # 첫 번째 은닉층에 ReLU 활성화 함수 적용
        x = torch.relu(self.fc1(x))
        # 두 번째 은닉층에 ReLU 활성화 함수 적용
        x = torch.relu(self.fc2(x))
        # 출력층에서 Q값 벡터 계산
        return self.fc3(x)


# DQN 에이전트 클래스
# Roll/Pitch 제어를 위한 이산 DQN 알고리즘을 구현한다.
# 상태: [ball_x, ball_y, ball_x_prev, ball_y_prev, roll_angle, pitch_angle] (6차원)
# 행동: bins_per_axis × bins_per_axis개의 이산 행동, 각 행동은 (roll 목표 각도, pitch 목표 각도) 쌍으로 맵핑
class DQNAgent:
    # state_dim: 상태 벡터 차원 (예: 6)
    # bins_per_axis: roll, pitch 이산화 단계 수 (예: 7)
    # gamma: 할인율 (예: 0.99)
    # lr: 학습률 (예: 1e-3)
    # batch_size: 미니배치 크기 (예: 64)
    # buffer_size: 리플레이 버퍼 크기 (예: 100000)
    # epsilon_start: ε-greedy 탐색 시 초기 ε 값 (예: 1.0)
    # epsilon_end: ε-greedy 탐색 시 최소 ε 값 (예: 0.05)
    # epsilon_decay: ε 감소 계수 (예: 0.995)
    # update_target_every: 타깃 네트워크를 업데이트하는 주기 (스텝 단위, 예: 1000)
    # device: 연산 디바이스 지정 (예: "cpu" 또는 "cuda")
    def __init__(
        self,
        state_dim=6,
        bins_per_axis=7,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=100000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        update_target_every=1000,
        device=None
    ):
        # 상태 및 행동 관련 설정
        self.state_dim = state_dim
        self.bins_per_axis = bins_per_axis
        # 행동 개수 = roll 이산화 개수 × pitch 이산화 개수
        self.action_dim = bins_per_axis * bins_per_axis
        # 강화학습 하이퍼파라미터 설정
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # ε-greedy 탐색 설정
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        # 타깃 네트워크 동기화 주기
        self.update_target_every = update_target_every
        # 연산 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 이산화된 roll/pitch 목표 각도 벡터 생성 ([-1.57, +1.57] 구간을 bins_per_axis 단계로 분할)
        self.pos_bins = np.linspace(-1.57, 1.57, bins_per_axis)

        # Q 네트워크와 타깃 네트워크 초기화
        self.q_network = DQNNetwork(state_dim, self.action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, self.action_dim).to(self.device)
        # 초기에는 두 네트워크가 동일하게 시작
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 타깃 네트워크는 학습 모드 비활성화

        # 최적화 알고리즘: Adam
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        # 리플레이 버퍼: 최대 buffer_size 크기의 데크(Deque)
        self.replay_buffer = deque(maxlen=self.buffer_size)
        # 학습 스텝 카운터 (타깃 네트워크 업데이트용)
        self.learn_step_counter = 0

    # 이산 행동 인덱스를 (roll 목표 각도, pitch 목표 각도)로 매핑
    # index: 0 ~ action_dim-1 사이 정수
    # 반환: (target_roll, target_pitch)
    def map_index_to_action(self, index):
        i_roll = index // self.bins_per_axis
        i_pitch = index % self.bins_per_axis
        target_roll = self.pos_bins[i_roll]
        target_pitch = self.pos_bins[i_pitch]
        return target_roll, target_pitch

    # ε-greedy 정책으로 행동 선택
    # state: 현재 상태 벡터 (6차원 리스트 또는 배열)
    # evaluate: 평가 모드이면 ε 탐색 없이 항상 greedy 선택
    # 반환: 행동 인덱스 (정수)
    def select_action(self, state, evaluate=False):
        # 탐색 모드 (evaluate=False)이고 랜덤 확률 < ε이면 랜덤 행동
        if (not evaluate) and (random.random() < self.epsilon):
            return random.randrange(self.action_dim)
        # 그렇지 않으면 Q 네트워크를 통해 Q값을 계산하고 argmax 행동 선택
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    # (state, action, reward, next_state, done) 튜플을 리플레이 버퍼에 저장
    # state: 현재 상태 벡터
    # action: 행동 인덱스 (정수)
    # reward: 보상 값 (float)
    # next_state: 다음 상태 벡터
    # done: 에피소드 종료 플래그 (0.0 또는 1.0)
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # 리플레이 버퍼에서 미니배치 랜덤 샘플링
    # 반환: (states, actions, rewards, next_states, dones) 텐서 묶음
    def sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # 넘파이 배열 → 파이토치 텐서로 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

    # 한 학습 스텝: 배치를 샘플링하여 손실(loss)을 계산하고 네트워크 파라미터를 갱신
    def train_step(self):
        # 버퍼에 학습할 만큼 충분한 샘플이 없으면 리턴
        if len(self.replay_buffer) < self.batch_size:
            return

        # 미니배치 샘플링
        states, actions, rewards, next_states, dones = self.sample_batch()

        # 현재 네트워크를 통해 Q(s, a) 계산
        # gather를 사용하여 각 상태별 선택된 행동에 해당하는 Q값을 가져옴
        q_pred = self.q_network(states).gather(1, actions)

        # 타깃 네트워크를 통해 다음 상태에서의 최대 Q값 계산 (학습 시에는 gradient 계산 중단)
        with torch.no_grad():
            next_q = self.target_network(next_states)
            max_next_q, _ = torch.max(next_q, dim=1, keepdim=True)
            # Bellman 타깃: r + γ * max_{a'} Q_target(s', a')
            q_target = rewards + (1.0 - dones) * self.gamma * max_next_q

        # 손실 함수: MSE (q_pred vs. q_target)
        loss = nn.MSELoss()(q_pred, q_target)

        # 역전파 및 최적화 수행
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε 감소 (탐색 비율 점진적 감소)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 타깃 네트워크 주기적 업데이트
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_every == 0:
            # 메인 네트워크 파라미터를 타깃 네트워크에 복사
            self.target_network.load_state_dict(self.q_network.state_dict())

    # Q 네트워크 파라미터 저장
    # filepath: 저장할 파일 경로 (예: "dqn_model.pth")
    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    # Q 네트워크 파라미터 로드
    # filepath: 로드할 파일 경로 (예: "dqn_model.pth")
    # 로드 이후 타깃 네트워크도 동일하게 업데이트
    def load(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
