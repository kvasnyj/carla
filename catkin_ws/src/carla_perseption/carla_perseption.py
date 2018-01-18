#!/usr/bin/env python

#~/carla/CarlaUE4.sh /Game/Maps/Town01 -carla-server -fps=15 -windowed -ResX=800 -ResY=600 

import rospy

import random
import time
import numpy as np
import cv2

import sys
sys.path.insert(0, '/home/kvasnyj/Dropbox/carla/')
from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from Line import Line

from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32
from styx_msgs.msg import CarState

class CarlaPerseption(object):
    h, w = None, None
    warp_src, warp_dst = [], []

    left_lane = Line()
    right_lane = Line()

    def __init__(self):
        rospy.logdebug("CarlaPerseption started")
        rospy.init_node('carla_perseption', log_level=rospy.DEBUG)

        self.carstate_pub = rospy.Publisher('carstate', CarState, queue_size=1)

        rospy.spin()

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

def run_carla_client(host, port):
    with make_carla_client(host, port) as client:
        rospy.logdebug('CarlaClient connected')
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=0,
            NumberOfPedestrians=0,
            WeatherId=1)  # random.choice([1, 3, 7, 8, 14]))
        settings.randomize_seeds()

        cameraSeg = Camera('CameraSemanticSegmentation', PostProcessing='SemanticSegmentation')
        cameraSeg.set_image_size(800, 600)
        cameraSeg.set_position(30, 0, 130)
        settings.add_sensor(cameraSeg)

        scene = client.load_settings(settings)

        number_of_player_starts = len(scene.player_start_spots)
        player_start = 1  # random.randint(0, max(0, number_of_player_starts - 1))

        client.start_episode(player_start)

        perseption = CarlaPerseption()

        while True:
            measurements, sensor_data = client.read_data()

            sem = sensor_data.get('CameraSemanticSegmentation').data
            carstate = CarState()
            carstate.position = perseption.process_image(sem)
            carstate.speed = measurements.player_measurements.forward_speed

            perseption.carstate_pub.publish(carstate)

def main():
    while True:
        try:
            run_carla_client('localhost', 2000)

            return

        except TCPConnectionError as error:
            rospy.logerr(error)
            time.sleep(1)
        except Exception as exception:
            rospy.logerr(exception)
            sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start carla_perseption node.')