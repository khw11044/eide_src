

빌드 

```
colcon build --symlink-install
```

```
source ~/rosa_ws/install/local_setup.bash

source install/setup.bash
```

### yolo 보기

yolo 실행하여 그 결과값 pub
```
ros2 run yolov11_ros yolov11_msg_publisher
```


yolo 화면 보기 
```
ros2 run yolov11_ros yolov11_ros_viewer
```

![Image](https://github.com/user-attachments/assets/328525fa-f668-4f3c-8e32-80b5add29f30)


## agent 실행

```
ros2 run edie_agent edie_agent
```

```
ros2 pkg list | grep edie_agent
```


### 키보드 원격 제어 

```
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/edie8/diff_drive_controller/cmd_vel_unstamped
```

### 에디 상태 및 모터 제어 GUI 

```
ros2 run edie8_test_gui edie8_test_gui
```

```
ros2 topic pub /edie8/diff_drive_controller/cmd_vel_unstamped geometry_msgs/msg/Twist "{linear: {x: 2.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```