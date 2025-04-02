

빌드 

```
colcon build --symlink-install
```

```
source ~/rosa_ws/install/local_setup.bash

source install/setup.bash
```

### yolo 보기

yolo 실행하여 그 결과값 확인 
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