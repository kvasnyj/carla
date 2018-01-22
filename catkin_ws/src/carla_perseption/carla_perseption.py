#!/usr/bin/env python

#~/carla/CarlaUE4.sh /Game/Maps/Town01 -carla-server -fps=15 -windowed -ResX=800 -ResY=600

import rospy
import time
import numpy as np
import cv2

from Line import Line

from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32
from styx_msgs.msg import CarState, CarControl

carcontrol_pub = None

class CarlaPerseption(object):
    h, w = None, None
    warp_src, warp_dst = [], []

    left_lane = Line()
    right_lane = Line()

    def define_warper(self):
        basex = 520
        width = 100
        height = 100

        src = np.float32([
            [0, basex],
            [790, basex],
            [215, 400],
            [580, 400]
        ])

        dst = np.float32([
            [(self.w-width)/2, basex],
            [self.w - (self.w-width)/2, basex],
            [(self.w-width)/2, basex-height],
            [self.w - (self.w-width)/2, basex-height]
        ])

        return src, dst

    def warper(self, img):
        # Compute and apply perpective transform
        M = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

    def sem2bin(self, img):
        bin = np.uint8(img == 7)*255

        bin = cv2.Canny(bin, 0, 255)

        return bin

    def define_position(self, warped):
        pts = np.argwhere(warped[:, :])
        position = self.w/2
        left  = np.mean(pts[(pts[:,1] < position) & (pts[:,0] > 410)][:,1])
        right = np.mean(pts[(pts[:,1] > position) & (pts[:,0] > 410)][:,1])
        position = -(right-self.w/2-40)-10

        return position

    def process_image(self, src_sem):
        if self.h == None: self.h = src_sem.shape[0]
        if self.w == None: self.w = src_sem.shape[1]
        if  len(self.warp_src) == 0: self.warp_src, self.warp_dst = self.define_warper()

        img = np.copy(src_sem)
        img = self.warper(img)
        img = self.sem2bin(img)
        position = self.define_position(img)

        return position

def carstate_cb(msg):
    msg.position = perseption.process_image(np.asarray(msg.camera1d).reshape(800, 600))

    car_control = CarControl()
    car_control.steer = 0
    car_control.throttle = 1

    carcontrol_pub.publish(car_control)


def main():
    perseption = CarlaPerseption()

    rospy.logdebug("CarlaPerseption started")
    rospy.init_node('carla_perseption', log_level=rospy.DEBUG)

    carcontrol_pub = rospy.Publisher('carstate', CarState, queue_size=1)
    rospy.Subscriber('/CarState', CarState, carstate_cb)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start carla_perseption node.')