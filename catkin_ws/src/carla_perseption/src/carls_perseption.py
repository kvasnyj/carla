import rospy

import random
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
import cv2

from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from Line import Line

class CarlaPerseption(object):
    def __init__(self):
        self.waypoints = None

        rospy.logdebug("CarlaPerseption started")
        rospy.init_node('carla_perseption', log_level=rospy.DEBUG)

        self.measurements_pub = rospy.Publisher('measurements', Measurements, queue_size=1)

        rospy.spin()

def main():
    while True:
        try:
            #run_carla_client('localhost', 2000)

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