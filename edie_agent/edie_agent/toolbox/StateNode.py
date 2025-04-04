# StateNode.py

import math
import time
import threading
import logging

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from yolo_perception.msg import DetectionArray

logger = logging.getLogger(__name__)

class CustomRobotNode(Node):
    def __init__(self):
        super().__init__("custom_robot_node")
        self.current_position = (0.0, 0.0, 0.0)
        self.odom_received = False
        self.deviance = 0
        self.target_x = None
        self.target_y = None
        self.target_yaw = None
        self.move_flag = False
        self.move_start_time = 0.0
        self.move_duration = 0.0
        self.cur_linear_x = 0.0
        self.cur_angular_z = 0.0

        self.odom_sub = self.create_subscription(
            Odometry, "/edie8/diff_drive_controller/odom", self.odom_callback, 10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist, "/edie8/diff_drive_controller/cmd_vel_unstamped", 10
        )
        self.detection_result = []
        self.yolo_sub = self.create_subscription(
            DetectionArray, "/detection_results", self.yolo_callback, 10
        )
        self.timer = self.create_timer(0.01, self.cmd_vel_timer_cb)

        self.get_logger().info("CustomRobotNode initialized.")
        logger.info("CustomRobotNode 생성됨")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        self.current_position = (x, y, yaw)
        self.odom_received = True

    def yolo_callback(self, msg):
        self.detection_result = msg.detections
        if self.detection_result:
            bb = self.detection_result[0].bounding_box
            center_x = (bb[0] + bb[2]) / 2
            self.deviance = 640 - center_x
        else:
            self.deviance = 0

    def set_target_position(self, x, y, yaw):
        self.target_x = x
        self.target_y = y
        self.target_yaw = yaw
        self.get_logger().info(f"목표 위치 설정: ({x:.2f}, {y:.2f}, {yaw:.2f})")

    def error_calculate(self) -> dict:
        """
        목표 위치( self.target_x, self.target_y )로부터의 거리/각도 오차 계산.
        """
        if self.target_x is None or self.target_y is None or self.target_yaw is None:
            return {"distance_error": 0.0, "angle_error": 0.0, "yaw_error": 0.0}

        x, y, yaw = self.current_position
        dx = self.target_x - x
        dy = self.target_y - y

        # 목표 이동 거리 계산
        distance = round(math.sqrt(dx**2 + dy**2), 4)

        # 목표 각도 계산
        target_angle = math.atan2(dy, dx)
        angle_error = (target_angle - yaw + 2 * math.pi) % (2 * math.pi)
        if angle_error > math.pi:
            angle_error -= 2 * math.pi
        # angle_error = round(angle_error, 6)

        # 목표 yaw 오차 계산 (-π ~ π 보정)
        # yaw_error = (self.target_yaw - yaw + 2 * math.pi) % (2 * math.pi)
        yaw_error = (self.target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        # yaw_error = round(yaw_error, 6)


        angle_error_deg = round(angle_error * 180 / math.pi, 2)
        yaw_error_deg = round(yaw_error * 180 / math.pi, 2)
        
        self.get_logger().info(f"distance_error: {distance}")
        self.get_logger().info(f"angle_error: {angle_error} -> {angle_error_deg}도")
        self.get_logger().info(f"yaw_error: {yaw_error} -> {yaw_error_deg}도")

        return {
            "distance_error": distance,
            "angle_error": yaw_error,
            "yaw_error": yaw_error
        }

    def yaw_error_calculate(self):
        if self.target_yaw is None:
            return {"yaw_error": 0.0}
        _, _, yaw = self.current_position
        yaw_error = (self.target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        return {"yaw_error": yaw_error}

    def publish_twist_to_cmd_vel(self, linear_x, angular_z, duration):
        self.cur_linear_x = linear_x
        self.cur_angular_z = angular_z
        self.move_duration = duration
        self.move_start_time = time.time()
        self.move_flag = True

    def cmd_vel_timer_cb(self):
        if self.move_flag:
            elapsed = time.time() - self.move_start_time
            if elapsed < self.move_duration:
                twist = Twist()
                twist.linear.x = self.cur_linear_x
                twist.angular.z = self.cur_angular_z
                self.cmd_vel_pub.publish(twist)
            else:
                self.cmd_vel_pub.publish(Twist())
                self.move_flag = False


# -------------------------------------------------
# ✅ 여기에 싱글턴 _get_robot_node 함수 정의!
# -------------------------------------------------
_robot_node: CustomRobotNode = None
_ros_thread = None
_initialized = False

def _get_robot_node() -> CustomRobotNode:
    global _robot_node, _ros_thread, _initialized
    if _robot_node is None:
        if not _initialized:
            logger.info("ROS2 초기화 시작")
            rclpy.init(args=None)
            _initialized = True

        _robot_node = CustomRobotNode()

        def ros_spin():
            try:
                rclpy.spin(_robot_node)
            except Exception as e:
                logger.error(f"ROS2 스핀 오류: {str(e)}")

        _ros_thread = threading.Thread(target=ros_spin, daemon=True)
        _ros_thread.start()

        timeout = 3.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if _robot_node.odom_received:
                logger.info("Odometry 데이터 수신 확인")
                break
            time.sleep(0.1)
        else:
            logger.warning("Odometry 데이터 수신 대기 시간 초과")

        logger.info("ROS2 노드 초기화 완료")

    return _robot_node
