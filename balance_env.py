from controller import Supervisor

class BalanceEnv:
    """Webots balance environment."""

    def __init__(self):
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        # 모터 디바이스 획득
        self.motors = [self.robot.getDevice(f'motor {i}') for i in range(1, 4)]
        # 위치 제어 모드 초기화
        for m in self.motors:
            m.setPosition(0.0)
        # 공 위치 정보 필드 획득
        ball_node = self.robot.getFromDef('Ball')
        self.translation_field = ball_node.getField('translation')
        self.counter = 0

    def _get_observation(self):
        return self.translation_field.getSFVec3f()

    def reset(self):
        self.robot.simulationReset()
        for m in self.motors:
            m.setPosition(0.0)
        self.counter = 0
        self.robot.step(self.time_step)
        return self._get_observation()

    def step(self, positions):
        for m, p in zip(self.motors, positions):
            m.setPosition(p)
        self.counter += 1
        self.robot.step(self.time_step)
        obs = self._get_observation()
        done = obs[2] < 0.09
        if done:
            self.robot.simulationReset()
        reward = 1.0 if not done else 0.0
        return obs, reward, done, {}
