#!/usr/bin/env python3
import time
import math
import logging
from geometry_msgs.msg import Twist
from langchain.agents import tool

from .StateNode import _get_robot_node
from .watch import yolo_tool  # 이미 @tool 붙어 있음

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------
# LangChain 도구 함수
# --------------------------------

@tool
def get_robot_pose() -> str:
    """
    로봇의 현재 위치를 반환.
    x, y, yaw (라디안 단위).
    """
    try:
        agent = _get_robot_node()
        x, y, yaw = agent.current_position
        if not agent.odom_received:
            logger.warning("Odometry 데이터 미수신")
            return "현재 위치: 데이터 없음 (odom 미수신)"
        return f"현재 위치: x={x:.2f}m, y={y:.2f}m, yaw={yaw:.2f}rad"
    except Exception as e:
        logger.error(f"위치 조회 오류: {str(e)}")
        return f"위치 조회 오류: {str(e)}"

@tool
def move_to_position(target_x: float, target_y: float, target_yaw: float) -> str:
    """
    로봇을 특정 위치까지 이동하기 위해 해당 도구를 사용합니다.
    현재 위치를 파악하고, 목표 위치로 방향을 target_yaw까지 회전한 이후 (target_x, target_y)까지 이동합니다.
    예: 몇 m 앞으로 전진해, 어디 위치로 이동해
    """
    agent = _get_robot_node()
    agent.set_target_position(target_x, target_y, target_yaw)
    x, y, yaw = agent.current_position
    print(f'current x, y, yaw: {x},{y},{yaw}')

    # 1) 목표 방향 제자리 회전
    while True:
        errors = agent.yaw_error_calculate()
        yaw_err = errors["yaw_error"]

        # 목표 yaw에 도달 판단 
        if abs(yaw_err) < 0.01:
            agent.move_flag = False
            agent.cmd_vel_pub.publish(Twist())  # 즉시 정지
            break
            
        # 기본 angular_speed 계산
        computed_speed = yaw_err # * 3

        # angular_speed의 절대값이 0보다 크고 2보다 작으면 최소 2로 설정
        if 0 < abs(computed_speed) < 4:
            computed_speed = 4.0 if computed_speed > 0 else -4.0

        angular_speed = max(-5.0, min(5.0, computed_speed))
        agent.publish_twist_to_cmd_vel(0.0, angular_speed, 0.05)
        time.sleep(0.05)

    # 2) 목표 위치까지 이동
    first_distance_err = None  # 최초의 distance_error를 저장할 변수
    while True:
        errors = agent.error_calculate()
        distance_err = errors["distance_error"]
        angle_err = errors["angle_error"]

        # 최초 distance_error 저장 (처음 루프에서 한 번)
        if first_distance_err is None:
            first_distance_err = distance_err

        # 만약 현재 distance_error가 초기값보다 5만큼 커지면 이동 중단
        if distance_err > first_distance_err + 2:
            agent.move_flag = False
            agent.cmd_vel_pub.publish(Twist())  # 즉시 정지
            return (f"이동 중단: 초기 distance_error {first_distance_err:.4f}보다 점점 더 커짐 (현재 {distance_err:.4f}) linear_speed를 반대로 해서 다시 할것")

        # 목표 위치 도달 판단 (예: 0.1m 이내)
        if distance_err < 0.1:
            agent.move_flag = False  # 이동 중지
            agent.cmd_vel_pub.publish(Twist())  # 즉시 정지
            return (f"이동 및 회전 완료: "
                    f"목표 위치 ({target_x:.2f}, {target_y:.2f}), "
                    f"목표 각도 {target_yaw:.2f} rad 도달.")
                    
        # 기본 angular_speed 계산
        computed_speed = angle_err # * 3

        # angular_speed의 절대값이 0보다 크고 2보다 작으면 최소 2로 설정
        if 0 < abs(computed_speed) < 4:
            computed_speed = 4.0 if computed_speed > 0 else -4.0

        # 후진 고려: 로봇의 헤딩과 목표 방향의 각차이가 90도(π/2:1.57)보다 크면, 목표가 후방에 있으므로 후진
        if abs(angle_err) > (math.pi / 2):
            linear_speed = -min(3.0, max(1.0, distance_err * 2))
            # 각속도도 부호 반전 (로봇이 목표 방향을 맞추도록)
            angular_speed = -max(-4.0, min(4.0, computed_speed))
        else:
            linear_speed = min(3.0, max(1.0, distance_err * 2))
            angular_speed = max(-4.0, min(4.0, computed_speed))

        # 방향을 더이상 고려할 필요 없는 경우 전진만 수행 
        if abs(angle_err) < 0.1:
            angular_speed = 0.0

        # 이동 명령 발행 (짧은 시간 동안 속도를 유지)
        agent.publish_twist_to_cmd_vel(linear_speed, angular_speed, 0.05)
        time.sleep(0.05)

@tool
def rotation_in_place(target_yaw: float) -> str:
    """
    제자리 회전을 해야하는 경우 해당 도구를 사용합니다.
    현재 위치를 파악하고, target_yaw까지 회전합니다.
    예: 제자리에서 20도 회전해
    """
    agent = _get_robot_node()
    agent.set_target_position(0.0, 0.0, target_yaw)

    while True:
        yaw_err = agent.error_calculate()["yaw_error"]
        if abs(yaw_err) < 0.01:
            agent.move_flag = False
            agent.cmd_vel_pub.publish(Twist())
            return f"회전 완료: {target_yaw:.2f} rad"

        speed = yaw_err
        if 0 < abs(speed) < 4:
            speed = 4.0 if speed > 0 else -4.0
        angular_speed = max(-5.0, min(5.0, speed))
        agent.publish_twist_to_cmd_vel(0.0, angular_speed, 0.1)
        time.sleep(0.1)

@tool
def face_detection(target_yaw: float) -> str:
    """
    예: 정면을 바라보게 회전하는 명령을 수행.
    deviance(픽셀 단위 편차)에 따라 회전값 조정.
    객체를 향해 제자리 회전할때 해당 도구를 사용합니다.
    """
    agent = _get_robot_node()
    while True:
        error = agent.deviance
        if abs(error) < 10:
            agent.move_flag = False
            agent.cmd_vel_pub.publish(Twist())
            return f"정면({target_yaw:.2f} rad) 바라봄"

        if 0 < abs(error) < 4:
            error = 4.0 if error > 0 else -4.0
        angular_speed = max(-5.0, min(5.0, error))
        agent.publish_twist_to_cmd_vel(0.0, angular_speed, 0.01)
        time.sleep(0.01)

@tool
def simple_move(linear_x: float, angular_z: float, duration: float) -> str:
    """
    간단한 이동 동작 도구 입니다. 
    선속도 (linear_x),
    각속도 (angular_z)
    를 제어하여 duration(초)동안 움직입니다. 
    전진 및 후진은 각속도 (angular_z)는 0.0이고 선속도 (linear_x)만 제어합니다.
    제자리 회전은 선속도 (linear_x)는 0.0이고 각속도 (angular_z)만 제어합니다.
    """
    
    agent = _get_robot_node()
    start = time.time()
    while True:
        if time.time() - start > duration:
            agent.move_flag = False
            agent.cmd_vel_pub.publish(Twist())
            return "이동 완료"
        if 0 < abs(linear_x) < 3:
            linear_x = 3.0 if linear_x > 0 else -3.0
        if 0 < abs(angular_z) < 4:
            angular_z = 4.0 if angular_z > 0 else -4.0
        agent.publish_twist_to_cmd_vel(linear_x, angular_z, 0.01)
        time.sleep(0.01)

# --------------------------------
# 테스트용 메인 함수
# --------------------------------
def main():
    agent = _get_robot_node()
    try:
        while rclpy.ok():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("종료 요청 수신됨")
    finally:
        if agent:
            agent.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
