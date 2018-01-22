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

from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32
from styx_msgs.msg import CarState, CarControl

class CarlaServer(object):
    def __init__(self, client):
        self.client = client
        self.carstate_pub = rospy.Publisher('carstate_source', CarState, queue_size=1)
        rospy.Subscriber('/carcontrol', CarControl, self.carcontrol_cb)

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

        scene = self.client.load_settings(settings)

        number_of_player_starts = len(scene.player_start_spots)
        player_start = 1  # random.randint(0, max(0, number_of_player_starts - 1))

        self.client.start_episode(player_start)
        rospy.loginfo("----start episode")


    def read_data(self):
        rospy.loginfo("----read data")
        measurements, sensor_data = self.client.read_data()

        sem = sensor_data.get('CameraSemanticSegmentation').data
        carstate = CarState()
        carstate.speed = measurements.player_measurements.forward_speed
        carstate.camera1d = np.asarray(sem).astype(int).flatten().tolist()

        self.carstate_pub.publish(carstate)


    def carcontrol_cb(self, msg):
        rospy.loginfo("----carcontrol_cb fired")
        if msg != None:
            rospy.logdebug("carcontrol_cb fired. steer: %s, throttle: %s", msg.steer, msg.throttle)
            self.client.send_control(
                steer=msg.steer,
                throttle=msg.throttle,
                brake=False,
                hand_brake=False,
                reverse=False)   
        
        self.read_data()     
          

def main():
    rospy.loginfo("CarlaServer started")
    rospy.init_node('carla_server')

    while not rospy.is_shutdown():
        try:
            with make_carla_client('localhost', 2000) as client:
                cs = CarlaServer(client)
                cs.carcontrol_cb(None)
                rospy.spin()
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
        rospy.logerr('Could not start carla_server node.')