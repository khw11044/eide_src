# action.py
import time

from langchain.agents import tool
from .StateNode import _get_robot_node

@tool
def action_ears(left_pos: float = 1.0, right_pos: float = 1.0) -> str:
    """
    왼쪽/오른쪽 귀의 위치를 제어합니다. 
    left_pos와 right_pos 값의 범위는 [0.0 - 1.0] 입니다.
    현재 상태는 StateNode 내부 변수로 저장됩니다.
    대화를 할때 귀를 움직입니다. 
    왼쪽 귀만 움직일때, right_pos는 0.0이고 left_pos의 값을 제어합니다. 
    오른쪽 귀만 움직일때, left_pos는 0.0이고 right_pos의 값을 제어합니다. 
    """
    agent = _get_robot_node()
    agent.left_ear_position = left_pos
    agent.right_ear_position = right_pos
    agent.publish_ear_position()

    return f"귀 위치 설정됨: 왼쪽={left_pos}, 오른쪽={right_pos}"

@tool
def reset_ears() -> str:
    """
    양쪽 귀를 기본 위치(0.0)로 되돌립니다.
    """
    agent = _get_robot_node()
    agent.left_ear_position = 0.0
    agent.right_ear_position = 0.0
    agent.publish_ear_position()
    return "귀를 기본 위치(0.0)로 리셋했습니다."

@tool
def wave_ears(count: int = 3, amplitude: float = 1.0, interval: float = 0.5) -> str:
    """
    귀를 교차로 흔듭니다.
    - count: 반복 횟수
    - amplitude: 귀의 최대 각도 Range [0.0 - 1.0]
    - interval: 귀를 바꿔 흔드는 시간 간격 (초)
    """
    agent = _get_robot_node()

    for i in range(count):
        # 왼쪽만 올림
        agent.left_ear_position = amplitude
        agent.right_ear_position = 0.0
        agent.publish_ear_position()
        time.sleep(interval)

        # 오른쪽만 올림
        agent.left_ear_position = 0.0
        agent.right_ear_position = amplitude
        agent.publish_ear_position()
        time.sleep(interval)

    # 마지막엔 양쪽 다 내림
    agent.left_ear_position = 0.0
    agent.right_ear_position = 0.0
    agent.publish_ear_position()

    return f"귀를 {count}회 교차로 흔들었습니다. (amplitude={amplitude}, interval={interval}s)"


@tool
def action_legs(left_pos: float = -0.02, right_pos: float = -0.02) -> str:
    """
    왼쪽/오른쪽 다리의 위치를 제어합니다.
    기본값은 0.0입니다.
    값은 [-0.02, -0.01, 0.0] 중 하나 입니다. 
    랜덤으로 값을 정합니다.
    """
    agent = _get_robot_node()
    agent.left_leg_position = left_pos
    agent.right_leg_position = right_pos
    agent.publish_leg_position()

    return f"다리 위치 설정됨: 왼쪽={left_pos}, 오른쪽={right_pos}"



@tool
def reset_legs() -> str:
    """양쪽 다리를 기본 위치(0.0)로 되돌립니다."""
    agent = _get_robot_node()
    agent.left_leg_position = 0.0
    agent.right_leg_position = 0.0
    agent.publish_leg_position()
    return "다리를 기본 위치(0.0)로 리셋했습니다."


@tool
def wave_legs(count: int = 3, amplitude: float = -0.02, interval: float = 0.5) -> str:
    """
    다리를 교차로 흔듭니다.
    - count: 반복 횟수
    - amplitude: 다리 움직이는 최대 위치, 값은 [-0.02, -0.01, 0.0] 중 하나 입니다. 
    - interval: 시간 간격
    """
    agent = _get_robot_node()

    for i in range(count):
        agent.left_leg_position = amplitude
        agent.right_leg_position = 0.0
        agent.publish_leg_position()
        time.sleep(interval)

        agent.left_leg_position = 0.0
        agent.right_leg_position = amplitude
        agent.publish_leg_position()
        time.sleep(interval)

    agent.left_leg_position = 0.0
    agent.right_leg_position = 0.0
    agent.publish_leg_position()
    return f"다리를 {count}회 교차로 흔들었습니다."


@tool
def action_legs_and_ears(l_leg: float = 0.0, r_leg: float = 0.0,
                         l_ear: float = 1.0, r_ear: float = 1.0) -> str:
    """
    다리와 귀를 동시에 원하는 위치로 설정합니다.
    다리의 값은 [-0.02, -0.01, 0.0] 중 하나 입니다.
    귀의 값의 범위는 [0.0 ~ 1.0] 입니다.
    """
    agent = _get_robot_node()

    agent.left_leg_position = l_leg
    agent.right_leg_position = r_leg
    agent.left_ear_position = l_ear
    agent.right_ear_position = r_ear

    agent.publish_leg_position()
    agent.publish_ear_position()

    return (f"다리 및 귀 위치 설정됨:\n"
            f"  다리: 왼쪽={l_leg}, 오른쪽={r_leg}\n"
            f"  귀:   왼쪽={l_ear}, 오른쪽={r_ear}")