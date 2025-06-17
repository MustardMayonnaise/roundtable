import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DQN_Agent import DQNAgent

def run_dqn(robot, time_step,
            motor1, ps1, motor3, ps3,
            translation_field,
            num_episodes, action_interval,
            episode_train_steps, z_threshold,
            model_path):

    NOT_MOVED = -0.1
    MOVE_REWARD_MIN = 0.5
    MOVE_REWARD_MAX = 2.0
    DROP_THE_BALL = -10.0
    TIME_REWARD_FACTOR = 0.02 * action_interval
    EPISODE_DONE_REWARD = 30.0
    MAX_STEPS_PER_EPISODE = 1000


    # ── 에이전트 초기화 ─────────────────────────────────────────────────────────
    agent = DQNAgent(
        state_dim=6,
        bins_per_axis=7,                # roll/pitch 이산 단계 수
        gamma=0.99, lr=1e-3, batch_size=256,
        buffer_size=1_000_000,          # 버퍼 1M으로 확장
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
        update_target_every=100
    )

    rewards_hist = []
    steps_per_episode_hist = []
    best_reward = -float("inf")
    episode = 0
    step_counter = 0

    # ── 메인 루프: num_episodes만큼 반복 ──────────────────────────────────────────
    while episode < num_episodes:
        # 에피소드 초기화
        step_in_ep = 0
        total_reward = 0.0
        ball_x_prev, ball_y_prev = 0.0, 0.0
        done = False

        # ── 하나의 에피소드 실행 ────────────────────────────────────────────────
        while not done and step_in_ep < MAX_STEPS_PER_EPISODE:
            # (1) Webots 시뮬레이션 한 스텝 전진
            if robot.step(time_step) == -1:
                return
            step_counter += 1
            step_in_ep += 1

            # (2) 현재 상태 관측
            bx, by, bz = translation_field.getSFVec3f()
            roll, pitch = ps1.getValue(), ps3.getValue()
            state = [bx, by, ball_x_prev, ball_y_prev, roll, pitch]

            # (3) action_interval 간격마다 이산 행동 선택 & 모터 명령
            action_idx = None
            if step_counter % action_interval == 0:
                action_idx = agent.select_action(state)                     # int index
                roll_cmd, pitch_cmd = agent.map_index_to_action(action_idx) # continuous 값
                motor1.setPosition(float(roll_cmd))
                motor3.setPosition(float(pitch_cmd))

            # (4) 모터 적용 후 다음 스텝 관측
            if robot.step(time_step) == -1:
                return
            step_counter += 1
            step_in_ep += 1

            bx2, by2, bz2 = translation_field.getSFVec3f()
            roll2, pitch2 = ps1.getValue(), ps3.getValue()
            next_state = [bx2, by2, bx, by, roll2, pitch2]

            # (5) 보상 계산
            prev_d = np.hypot(ball_x_prev, ball_y_prev)
            cur_d  = np.hypot(bx2, by2)
            delta  = prev_d - cur_d
            r_pos = NOT_MOVED if abs(delta) < 1e-6 else float(
                        np.clip(delta, MOVE_REWARD_MIN, MOVE_REWARD_MAX)
                    )
            r_time = TIME_REWARD_FACTOR * action_interval
            r_fall = DROP_THE_BALL if bz2 < z_threshold else 0.0

            reward = r_time + r_pos + r_fall

            # (6) 최대 스텝 도달 시 추가 보상
            if step_in_ep >= MAX_STEPS_PER_EPISODE:
                reward += EPISODE_DONE_REWARD

            total_reward += reward
            done = (bz2 < z_threshold)

            # (7) transition 저장 및 학습
            if action_idx is not None:
                agent.store_transition(state, action_idx, reward, next_state, float(done))
                agent.train_step()

            # (8) 상태 업데이트
            ball_x_prev, ball_y_prev = bx2, by2

        # ── 에피소드 종료 처리 ───────────────────────────────────────────────────
        rewards_hist.append(total_reward)
        steps_per_episode_hist.append(step_in_ep)

        # (9) 최고 모델 저장
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(model_path)
            print(f"[DQN] New best {best_reward:.2f}, saved to {model_path}")

        # (10) 에피소드 후 추가 학습
        for _ in range(episode_train_steps):
            agent.train_step()

        episode += 1
        robot.simulationReset()
        ps1.enable(time_step); ps3.enable(time_step)
        print(f"[DQN] Episode {episode} 종료: steps={step_in_ep}, reward={total_reward:.2f}")

    # ── 학습 결과 시각화 ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        'Episode': range(1, len(rewards_hist) + 1),
        'Total_Reward': rewards_hist,
        'Steps_in_Episode': steps_per_episode_hist
    })
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Episode vs Total Reward
    axs[0].plot(df['Episode'], df['Total_Reward'], marker='.', linestyle='-')
    axs[0].set_title('Episode vs Total Reward');    axs[0].grid(True)
    # Episode vs Steps
    axs[1].plot(df['Episode'], df['Steps_in_Episode'], marker='.', linestyle='-')
    axs[1].set_title('Episode vs Steps');          axs[1].grid(True)
    # Reward vs Steps
    axs[2].scatter(df['Total_Reward'], df['Steps_in_Episode'], alpha=0.6)
    axs[2].set_title('Reward vs Steps');           axs[2].grid(True)
    plt.tight_layout()
    plt.savefig("DQN_summary.png")
    plt.show()
