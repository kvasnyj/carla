#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32
from styx_msgs.msg import CarState, CarControl

# rosrun rqt_reconfigure rqt_reconfigure
Kp = 0.008
Ki = 0.0001
Kd = 0.0

class CarlaControl(object):
    def __init__(self):
        self.prev_cte = 0
        self.int_cte = 0

        rospy.set_param('PID', {'Kp': Kp, 'Ki': Ki, 'Kd': Kd})

    def update_error(self, cte):
        diff_cte = cte - self.prev_cte
        self.prev_cte = cte
        self.int_cte += cte

        gains = rospy.get_param('PID')
        Kp, Ki, Kd = gains['Kp'], gains['Ki'], gains['Kd']

        steer = -Kp * cte - Kd * diff_cte - Ki * self.int_cte
        if steer > 0.3: steer = 0.3
        if steer < -0.3: steer = -0.3

        rospy.loginfo("cte: %s, steer: %s", cte, steer)

        return steer


def carstate_cb(msg):
    control_msg = CarControl()
    control_msg.steer = car_control.update_error(msg.position)

    control_msg.throttle = 1;
    if  (msg.speed >= 30): control_msg.throttle = 0

    carscontrol_pub.publish(control_msg)

if __name__ == '__main__':
    try:
        car_control = CarlaControl()
        rospy.init_node('carla_control')
        rospy.loginfo("carla_control started")

        carscontrol_pub = rospy.Publisher('carcontrol', CarControl, queue_size=1)
        rospy.Subscriber('/carstate_planning', CarState, carstate_cb)
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start carla_control node.')