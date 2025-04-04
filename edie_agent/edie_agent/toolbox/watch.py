# watch.py

from langchain.agents import tool
from .StateNode import _get_robot_node

@tool
def yolo_tool():
    """
    카메라 피드를 기반으로 객체를 감지합니다.
    deviance는 로봇 기준 편차값(640픽셀 - 감지된 x중심).
    """
    agent = _get_robot_node()
    results = []
    for detection in agent.detection_result:
        data = {
            "label": detection.label,
            "confidence": detection.confidence,
            "bounding_box": list(detection.bounding_box),
            "width": detection.width,
            "height": detection.height,
        }
        results.append(data)

    return results
