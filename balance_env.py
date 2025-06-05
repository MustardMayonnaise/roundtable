"""Webots 환경 관리 모듈.

이 모듈은 Supervisor 컨트롤러로 동작하여 공 위치와 모터 상태를
관측하고, 강화학습 에이전트와의 인터페이스를 제공합니다.
"""

from controller import Supervisor


class BalanceBotEnv:
    def __init__(self, max_steps=1000):
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        # 모터 디바이스 설정 (motor2는 사용하지 않음)
        self.motor1 = self.robot.getDevice('motor 1')
        self.motor2 = self.robot.getDevice('motor 2')
        self.motor3 = self.robot.getDevice('motor 3')

        # 위치 제어 비활성화 후 속도 제어용으로 설정
        for m in (self.motor1, self.motor2, self.motor3):
            m.setPosition(float('inf'))
            m.setVelocity(0.0)

        # 공 노드
        self.ball_node = self.robot.getFromDef('Ball')
        self.translation_field = self.ball_node.getField('translation')

        # 모터 각도 필드
        self.joint1_sensor = self.motor1.getPositionSensor()
        self.joint3_sensor = self.motor3.getPositionSensor()
        for s in (self.joint1_sensor, self.joint3_sensor):
            s.enable(self.time_step)

        self.prev_ball_pos = [0.0, 0.0]
        self.max_steps = max_steps
        self.step_count = 0

    def get_state(self):
        pos = self.translation_field.getSFVec3f()
        x, y = pos[0], pos[1]
        vx = (x - self.prev_ball_pos[0]) / (self.time_step / 1000.0)
        vy = (y - self.prev_ball_pos[1]) / (self.time_step / 1000.0)

        self.prev_ball_pos = [x, y]

        theta1 = self.joint1_sensor.getValue()
        theta3 = self.joint3_sensor.getValue()

        return [x, y, vx, vy, theta1, theta3]

    def compute_reward(self, state):
        # 간단한 보상 설계
        x, y, vx, vy, _, _ = state
        alpha = 1.0
        beta = 0.1
        gamma = 1.0
        distance_penalty = -alpha * (x ** 2 + y ** 2)
        alive_bonus = beta
        static_penalty = -gamma if abs(vx) < 1e-3 and abs(vy) < 1e-3 else 0.0
        return distance_penalty + alive_bonus + static_penalty

    def step(self, action):
        """액션은 [v1, v3] 형태의 모터 속도 리스트."""
        self.motor1.setVelocity(action[0])
        self.motor3.setVelocity(action[1])

        self.robot.step(self.time_step)

        state = self.get_state()
        reward = self.compute_reward(state)

        done = self.translation_field.getSFVec3f()[2] < 0.09
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        return state, reward, done

    def reset(self):
        self.robot.simulationReset()
        self.robot.step(self.time_step)
        pos = self.translation_field.getSFVec3f()
        self.prev_ball_pos = [pos[0], pos[1]]
        self.step_count = 0
        return self.get_state()

