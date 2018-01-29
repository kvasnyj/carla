# Carla
My playground with Carla - open-source simulator for autonomous driving research, developed in cooperation by Intel Labs, Toyota Research Institute and the Computer Vision Center . http://carla.org

## Demo video
[![Turn right by PID controller](https://img.youtube.com/vi/mCuxJ2wRP04/0.jpg)](https://youtu.be/mCuxJ2wRP04)

## Modules
	1. Sever: get sensors data and send control packet
	2. Perception > Detection > Lanes detection
	3. Planning: following right lane
	4. Control: PID controller
	
![rosgraph](https://github.com/kvasnyj/carla/blob/master/catkin_ws/rosgraph.png "Rosgraph")

## Parameters
rosparam get / rosparam set
* PID/Kp
* PID/Pi
* PID/Kd

## RViz
rosrun rviz rviz

![rviz](https://github.com/kvasnyj/carla/blob/master/catkin_ws/rviz.png "RViz")
