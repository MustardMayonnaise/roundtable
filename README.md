# Roundtable Webots 트레이닝 가이드

이 저장소는 Webots 시뮬레이터에서 강화학습 에이전트를 트레이닝하기 위한 예제 코드를 포함합니다. `train.py`를 실행하여 에이전트를 학습시킬 수 있으며, 실행 시 사용할 에이전트를 선택할 수 있습니다.

## 실행 방법

1. Python 및 주요 의존성을 설치합니다. 필요한 주요 패키지는 다음과 같습니다.
   - `torch` (PyTorch)
   - `numpy`
   - `webots` Python API

2. Webots가 설치되어 있고 환경 변수가 설정되어 있다고 가정합니다. 이후 다음 명령으로 학습을 시작합니다.

```bash
python train.py --agent rqn  # RQNAgent 사용
# 또는
python train.py --agent sac  # SACAgent 사용
```

`--agent` 옵션을 통해 두 가지 에이전트 중 하나를 선택할 수 있습니다. 각 에이전트는 `RL_Agent.py`에 정의되어 있습니다.

## 참고 사항

- Webots에서 world 파일을 열고 시뮬레이션이 시작된 상태에서 위의 명령을 실행하면 학습 루프가 수행됩니다.
- GPU 사용을 위해서는 PyTorch 설치 시 CUDA 지원 버전을 사용해야 합니다.
