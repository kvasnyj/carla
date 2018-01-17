import rospy

#import random
#import time
import numpy as np
import cv2
import sys
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
    h, w = None, None
    warp_src, warp_dst = [], []

    left_lane = Line()
    right_lane = Line()

    def __init__(self):
        self.carstate = None

        rospy.logdebug("CarlaPerseption started")
        rospy.init_node('carla_perseption', log_level=rospy.DEBUG)

        self.carstate_pub = rospy.Publisher('carstate', CarState, queue_size=1)

        rospy.spin()

    def define_warper():
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
            [(w-width)/2, basex],
            [w - (w-width)/2, basex],
            [(w-width)/2, basex-height],
            [w - (w-width)/2, basex-height]
        ])

        return src, dst

    def warper(img):
        # Compute and apply perpective transform
        M = cv2.getPerspectiveTransform(warp_src, warp_dst)
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image

        return warped

    def sem2bin(img):
        bin = np.uint8(img == 7)*255

        bin = cv2.Canny(bin, 0, 255)

        return bin

    def define_position(warped)
        pts = np.argwhere(warped[:, :])
        position = w/2
        left  = np.mean(pts[(pts[:,1] < position) & (pts[:,0] > 410)][:,1])
        right = np.mean(pts[(pts[:,1] > position) & (pts[:,0] > 410)][:,1])
        position = -(right-w/2-40)-10

        return position

    def process_image(src_sem):
        if h == None: h = src_sem.shape[0]
        if w == None: w = src_sem.shape[1]
        if  len(warp_src) == 0: warp_src, warp_dst = define_warper()

        img = np.copy(src_sem)
        img = warper(img)
        img = sem2bin(img)
        position = define_position(img)

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

        while (true):
            measurements, sensor_data = client.read_data()

            sem = sensor_data.get('CameraSemanticSegmentation').data
            self.carstate = new CarState()
            self.carstate.position = perseption.process_image(sem)
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
        main()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')