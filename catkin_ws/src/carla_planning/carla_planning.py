#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32
from styx_msgs.msg import CarState, CarControl

def carstate_cb(msg):
    # Nothing change, planning to go by position
    msg.position = msg.position
    carstate_pub.publish(msg)

if __name__ == '__main__':
    try:
        rospy.init_node('carla_planning')
        rospy.loginfo("carla_planning started")

        carstate_pub = rospy.Publisher('carstate_planning', CarState, queue_size=1)
        rospy.Subscriber('/carstate_perseption', CarState, carstate_cb)
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start carla_planning node.')