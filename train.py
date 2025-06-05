from balance_env import BalanceEnv
from RL_Agent import RQNAgent


def main():
    env = BalanceEnv()
    state_dim = 3  # 공 위치 xyz
    action_dim = 27  # 3 motors each 3 candidates
    agent = RQNAgent(state_dim, action_dim)

    num_episodes = 10
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action_idx = agent.select_action(state)
            action = agent.map_discrete_to_velocity(action_idx)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action_idx, reward, next_state, done)
            agent.train()
            state = next_state
        print(f"Episode {ep} finished")


if __name__ == '__main__':
    main()
