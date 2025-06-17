import numpy as np
from DQN_Agent import DQNAgent
from SAC_Agent import SACAgent

def run_validation(robot, time_step,
                   motor1, ps1, motor3, ps3,
                   translation_field,
                   num_episodes, action_interval,
                   z_threshold, model_path,
                   target):
    """
    Validation (playback) routine:
      :param target: "dqn" or "sac" (agent type)
    """
    # 1) 에이전트 초기화 및 모델 로드
    if target == "dqn":
        agent = DQNAgent(
            state_dim=6, bins_per_axis=7,
            gamma=0.99, lr=1e-3, batch_size=64,
            buffer_size=100000,
            epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0,
            update_target_every=1000
        )
        agent.load(model_path)
        print(f"[VAL] Loaded DQN model: {model_path}")
        is_dqn = True
    elif target == "sac":
        agent = SACAgent(
            state_dim=6, action_dim=2,
            hidden_sizes=(256,256),
            gamma=0.99, lr=3e-4,
            replay_size=100000, batch_size=256,
            tau=0.005, alpha=0.2
        )
        agent.load(model_path)
        print(f"[VAL] Loaded SAC model: {model_path}")
        is_dqn = False
    else:
        raise ValueError(f"Unknown target for validation: {target}")

    # 2) 시뮬레이션 재생
    print(f"[VAL] Starting playback for {num_episodes} episodes...")
    episode = 0
    step_counter = 0
    ball_x_prev, ball_y_prev = 0.0, 0.0

    while episode < num_episodes:
        # 시뮬레이션 스텝
        if robot.step(time_step) == -1:
            break
        step_counter += 1

        # 행동 선택 및 적용 (지정된 간격마다)
        if step_counter % action_interval == 0:
            bx, by, bz = translation_field.getSFVec3f()
            roll_val = ps1.getValue()
            pitch_val = ps3.getValue()
            state = [bx, by, ball_x_prev, ball_y_prev, roll_val, pitch_val]

            if is_dqn:
                idx = agent.select_action(state, evaluate=True)
                target_roll, target_pitch = agent.map_index_to_action(idx)
            else:
                action = agent.select_action(state, evaluate=True)
                target_roll, target_pitch = float(action[0]), float(action[1])

            motor1.setPosition(target_roll)
            motor3.setPosition(target_pitch)
            ball_x_prev, ball_y_prev = bx, by

            # 공 낙하 확인
            if bz < z_threshold:
                episode += 1
                print(f"[VAL] Episode {episode} completed (bz={bz:.3f}). Resetting simulation.")
                robot.simulationReset()
                ps1.enable(time_step)
                ps3.enable(time_step)
                step_counter = 0
                ball_x_prev, ball_y_prev = 0.0, 0.0

    print("[VAL] Playback finished.")
