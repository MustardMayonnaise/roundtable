"""강화학습 학습 루프 스크립트.

Webots 환경을 초기화하고 RQN 또는 SAC 에이전트를 사용해
공 균형 제어 정책을 학습한다.
"""

import argparse

from balance_env import BalanceBotEnv
from RL_Agent import RQNAgent, SACAgent


def train(agent_type="rqn", episodes=10):
    env = BalanceBotEnv()
    state_dim = 6

    if agent_type == "rqn":
        action_dim = 9
        agent = RQNAgent(state_dim, action_dim)
        action_mapper = agent.map_discrete_to_velocity
    else:
        action_dim = 2
        agent = SACAgent(state_dim, action_dim)
        action_mapper = lambda a: a

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if agent_type == "rqn":
                action_idx = agent.select_action(state)
                action = action_mapper(action_idx)
            else:
                action = agent.select_action(state)

            next_state, reward, done = env.step(action)

            if agent_type == "rqn":
                agent.store_transition(state, action_idx, reward, next_state, done)
            else:
                agent.store_transition(state, action, reward, next_state, done)

            agent.train()
            state = next_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["rqn", "sac"], default="rqn")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    train(agent_type=args.agent, episodes=args.episodes)

