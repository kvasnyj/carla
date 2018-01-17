import rospy

#import random
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
import cv2

sys.path.insert(0, '/carla')
from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from Line import Line

from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import float32
from styx_msgs.msg import CarState

class CarlaPerseption(object):
    def __init__(self):
        self.carstate = None

        rospy.logdebug("CarlaPerseption started")
        rospy.init_node('carla_perseption', log_level=rospy.DEBUG)

        self.carstate_pub = rospy.Publisher('carstate', CarState, queue_size=1)

        rospy.spin()

def run_carla_client(host, port):
    global frame
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

        while (true):
            measurements, sensor_data = client.read_data()

            sem = sensor_data.get('CameraSemanticSegmentation').data
            self.carstate = new CarState()
            self.carstate.position = process_image(sem, img)
            self.carstate.speed = measurements.player_measurements.forward_speed

            self.carstate_pub.publish(carstate)

def main():
    while True:
        try:
            run_carla_client('localhost', 2000)

            print('Done.')
            return

        except TCPConnectionError as error:
            rospy.logerr(error)
            time.sleep(1)
        except Exception as exception:
            rospy.logerr(exception)
            sys.exit(1)

if __name__ == '__main__':
    try:
        CarlaPerseption()

        left_lane = Line()
        right_lane = Line()

        main()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')