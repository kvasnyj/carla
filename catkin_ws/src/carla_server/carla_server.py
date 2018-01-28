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
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo

class CarlaServer(object):
    def __init__(self, client):
        self.client = client
        self.carstate_pub = rospy.Publisher('carstate_source', CarState, queue_size=10)
        self.marker_pub = rospy.Publisher('car_marker', Marker, queue_size=10)
        self.image_rgb_pub = rospy.Publisher('car_image_rgb', Image, queue_size=10)
        self.image_sem_pub = rospy.Publisher('car_sem_rgb', Image, queue_size=10)

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

        cameraRGB = Camera('CameraRGB')
        cameraRGB.set_image_size(800, 600)
        cameraRGB.set_position(30, 0, 130)
        settings.add_sensor(cameraRGB)       

        scene = self.client.load_settings(settings)

        number_of_player_starts = len(scene.player_start_spots)
        player_start = 1  # random.randint(0, max(0, number_of_player_starts - 1))

        self.client.start_episode(player_start)


    def read_data(self):
        measurements, sensor_data = self.client.read_data()

        sem = sensor_data.get('CameraSemanticSegmentation').data
        self.pub_rviz(measurements, sensor_data)

        carstate = CarState()
        carstate.speed = measurements.player_measurements.forward_speed
        carstate.camera1d = np.asarray(sem).astype(int).flatten().tolist()
        self.carstate_pub.publish(carstate)

    def pub_rviz(self, measurements, sensor_data):
        player_measurements = measurements.player_measurements

        rospy.loginfo("pos: %s, %s", player_measurements.transform.location.x, player_measurements.transform.location.y)

        marker = Marker()
        marker.header.frame_id = "my_frame"
        marker.header.stamp    = rospy.get_rostime()
        marker.ns = "carla"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD;
        marker.pose.position.x = (player_measurements.transform.location.x - 29859)/1000
        marker.pose.position.y = (player_measurements.transform.location.y - 13317)/1000
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = player_measurements.transform.orientation.x;
        marker.pose.orientation.y = player_measurements.transform.orientation.y;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 1.0; 
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;        
        self.marker_pub.publish(marker)

        # http://docs.ros.org/api/sensor_msgs/html/msg/Image.html
        image = sensor_data.get('CameraRGB').data
        image_rgb = Image()
        image_rgb.encoding = 'bgr8'
        image_rgb.height = image.shape[0]
        image_rgb.width = image.shape[1]
        image_rgb.step = image.shape[1] * 3
        image_rgb.data = image.tostring()
        self.image_rgb_pub.publish(image_rgb)

        image_num = sensor_data.get('CameraSemanticSegmentation').data*20
        image_sem = Image()
        image_sem.encoding = 'mono8' # http://docs.ros.org/jade/api/sensor_msgs/html/image__encodings_8h_source.html
        image_sem.height = image_num.shape[0]
        image_sem.width = image_num.shape[1]
        image_sem.step = image_num.shape[1] * 3
        image_sem.data = image_num.tostring()
        self.image_sem_pub.publish(image_sem)

    def carcontrol_cb(self, msg):
        if msg != None:
            #rospy.loginfo("carcontrol_cb fired. steer: %s, throttle: %s", msg.steer, msg.throttle)
            self.client.send_control(
                steer = msg.steer,
                throttle = msg.throttle,
                brake = False,
                hand_brake = False,
                reverse = False) 
        
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