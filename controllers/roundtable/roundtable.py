from controller import Supervisor
from roundtable_DQN import run_dqn
from roundtable_SAC import run_sac
from roundtable_VAL import run_validation

# 설정 값 (원래대로 사용)
MODE = "val"                 # "train", "val"
TARGET = "dqn"                 # "dqn", "sac"
NUM_EPISODES = 1000            # 학습/검증 에피소드 수
ACTION_INTERVAL = 1            # 행동 선택 주기 (스텝 단위)
EPISODE_TRAIN_STEPS = 20       # 학습 모드 추가 학습 반복 횟수
Z_THRESHOLD = 0.09             # 공 낙하로 간주할 z 임계값
MODEL_PATH = "best_model_DQN.pth" if TARGET == "dqn" else "best_model_SAC.pth"

class RoundtableController(Supervisor):
    def __init__(self):
        super().__init__()
        # Webots 기본 타임스텝
        self.time_step = int(self.getBasicTimeStep())
        # 모터 및 센서 초기화
        self.motor1 = self._init_motor('motor 1')
        self.ps1    = self.motor1.getPositionSensor()
        self.ps1.enable(self.time_step)

        self.motor3 = self._init_motor('motor 3')
        self.ps3    = self.motor3.getPositionSensor()
        self.ps3.enable(self.time_step)

        # 공 노드 및 위치 필드
        ball_node = self.getFromDef('Ball')
        if ball_node is None:
            raise RuntimeError("Ball DEF 노드를 찾을 수 없습니다.")
        self.translation_field = ball_node.getField('translation')

    def _init_motor(self, name):
        m = self.getDevice(name)
        m.setPosition(float('inf'))
        m.setVelocity(1.0)
        return m

    def run(self):
        if MODE == "train":
            self._train()
        elif MODE == "val":
            self._validate()
        else:
            raise ValueError(f"MODE 값이 올바르지 않습니다: {MODE}")

    def _train(self):
        if TARGET == "dqn":
            run_dqn(
                self, self.time_step,
                self.motor1, self.ps1, self.motor3, self.ps3,
                self.translation_field,
                NUM_EPISODES,
                ACTION_INTERVAL,
                EPISODE_TRAIN_STEPS,
                Z_THRESHOLD,
                MODEL_PATH
            )
        else:  # sac
            run_sac(
                self, self.time_step,
                self.motor1, self.ps1, self.motor3, self.ps3,
                self.translation_field,
                NUM_EPISODES,
                ACTION_INTERVAL,
                EPISODE_TRAIN_STEPS,
                Z_THRESHOLD,
                MODEL_PATH
            )

    def _validate(self):
        # run_validation의 시그니처 그대로 사용 :contentReference[oaicite:0]{index=0}
        run_validation(
            self, self.time_step,
            self.motor1, self.ps1, self.motor3, self.ps3,
            self.translation_field,
            NUM_EPISODES,
            ACTION_INTERVAL,
            Z_THRESHOLD,
            MODEL_PATH,
            TARGET
        )

if __name__ == "__main__":
    controller = RoundtableController()
    controller.run()