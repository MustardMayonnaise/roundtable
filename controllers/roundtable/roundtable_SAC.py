import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SAC_Agent import SACAgent


def run_sac(robot, time_step,
            motor1, ps1, motor3, ps3,
            translation_field,
            num_episodes, action_interval,
            episode_train_steps, z_threshold,
            model_path):

    # 보상 파라미터 설정
    NOT_MOVED = -0.1
    MOVE_REWARD_MIN = 0.5
    MOVE_REWARD_MAX = 2.0
    DROP_THE_BALL = -10.0
    TIME_REWARD = 0.02 * action_interval
    MAX_STEPS_PER_EPISODE = 1000
    EPISODE_DONE_REWARD = 30.0
    #TIME_REWARD = 0

    agent = SACAgent(
        state_dim=6, action_dim=2,
        hidden_sizes=(256, 256),
        gamma=0.99, lr=3e-4,
        replay_size=1000000, batch_size=256,
        tau=0.005, alpha=0.2
    )

    best_reward = -float('inf')
    episode = 0
    step_counter = 0
    step_in_ep = 0
    total_reward = 0.0
    ball_x_prev, ball_y_prev = 0.0, 0.0

    # --- 데이터 수집 리스트 수정 ---
    rewards_hist = []
    steps_per_episode_hist = []

    while episode < num_episodes:
        if robot.step(time_step) == -1:
            break
        step_counter += 1
        step_in_ep += 1

        # 현재 상태 관측
        bx, by, bz = translation_field.getSFVec3f()
        roll, pitch = ps1.getValue(), ps3.getValue()

        # action_interval 마다 행동 선택 및 적용
        if step_counter % action_interval == 0:
            state = [bx, by, ball_x_prev, ball_y_prev, roll, pitch]
            action = agent.select_action(state, evaluate=False)
            motor1.setPosition(float(action[0]))
            motor3.setPosition(float(action[1]))

            # 다음 상태 관측 (실제 적용 후)
            if robot.step(time_step) == -1:
                break
            step_counter += 1
            step_in_ep += 1

            bx2, by2, bz2 = translation_field.getSFVec3f()
            roll2, pitch2 = ps1.getValue(), ps3.getValue()

            # reward 계산
            prev_d = np.hypot(ball_x_prev, ball_y_prev)
            cur_d = np.hypot(bx2, by2)
            delta = prev_d - cur_d
            r_pos = NOT_MOVED if abs(delta) < 1e-6 else np.clip(delta, MOVE_REWARD_MIN, MOVE_REWARD_MAX)
            r_time = TIME_REWARD
            r_fall = DROP_THE_BALL if bz2 < z_threshold else 0.0
            reward = r_time + r_pos + r_fall
            total_reward += reward

            done = (bz2 < z_threshold)
            next_state = [bx2, by2, bx, by, roll2, pitch2]
            agent.store_transition(state, action, reward, next_state, float(done))
            agent.train_step()
            ball_x_prev, ball_y_prev = bx2, by2

            if done or step_in_ep >= MAX_STEPS_PER_EPISODE:
                if step_in_ep >= MAX_STEPS_PER_EPISODE:
                    reward += EPISODE_DONE_REWARD
                    print(f"[SAC] Episode {episode} 종료(최대 스텝 도달): steps={step_in_ep}, reward={total_reward:.2f}")
                else:
                    print(f"[SAC] Episode {episode} 종료(낙구): steps={step_in_ep}, reward={total_reward:.2f}")

                rewards_hist.append(total_reward)
                steps_per_episode_hist.append(step_in_ep)

                if total_reward > best_reward:
                    best_reward = total_reward
                    agent.save(model_path)
                    print(f"[SAC] New best {best_reward:.2f}, saved to {model_path}")

                for _ in range(episode_train_steps):
                    agent.train_step()
                episode += 1
                step_in_ep = 0
                total_reward = 0.0
                ball_x_prev, ball_y_prev = 0.0, 0.0
                robot.simulationReset()
                ps1.enable(time_step)
                ps3.enable(time_step)

    # 에피소드, 보상, 스텝 수를 DataFrame으로 통합
    df = pd.DataFrame({
        'Episode': range(1, len(rewards_hist) + 1),
        'Total_Reward': rewards_hist,
        'Steps_in_Episode': steps_per_episode_hist
    })
    # --- 한 화면에 3개 그래프(subplots) ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # 1) 에피소드 vs 보상
    axs[0].plot(df['Episode'], df['Total_Reward'], marker='.', linestyle='-')
    axs[0].set_title('Episode vs Total Reward')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].grid(True)
    # 2) 에피소드 vs 스텝 수
    axs[1].plot(df['Episode'], df['Steps_in_Episode'], marker='.', linestyle='-')
    axs[1].set_title('Episode vs Steps per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps per Episode')
    axs[1].grid(True)
    # 3) 보상 vs 스텝 수
    axs[2].scatter(df['Total_Reward'], df['Steps_in_Episode'], alpha=0.6)
    axs[2].set_title('Total Reward vs Steps per Episode')
    axs[2].set_xlabel('Total Reward')
    axs[2].set_ylabel('Steps per Episode')
    axs[2].grid(True)
    plt.tight_layout()
    plt.savefig("SAC_summary.png")  # 한 번만 저장
    plt.show()