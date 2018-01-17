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

        # bin[0:350, :] = 0

        return bin

    def peaks_histogram(img, debug = False):
        left_fitx, right_fitx, left_fity, right_fity = [], [], [], []
        past_left, past_right = 0, w / 2

        topx = 390
        offset = 360

        for i in range(0, topx):
            histogram = np.sum(img[topx-i:h-i, :], axis=0)

            x_left = np.argmax(histogram[offset: int(w / 2)])+offset
            if (past_left == 0) | (abs(x_left-past_left)<150) & (histogram[x_left]>=5):
                left_fitx.append(x_left)
                left_fity.append(h-i)
                past_left = x_left

            x_right = int(w / 2 + np.argmax(histogram[int(w / 2):w]))
            if (past_right == w / 2) | (abs(x_right-past_right)<150) & (histogram[x_right]>=5):
                right_fitx.append(x_right)
                right_fity.append(h-i)
                past_right = x_right

        return left_fitx, left_fity, right_fitx, right_fity

    def curvature(leftx, lefty, rightx, righty, debug = False):
        leftx = np.float32(leftx)
        rightx = np.float32(rightx)
        lefty = np.float32(lefty)
        righty = np.float32(righty)

        yvals = np.arange(h - h / 2, h, 1.0)
        left_curverad, right_curverad = 0, 0

        left_fit, right_fit = None, None
        if len(lefty)>5:
            y_eval = np.max(lefty)
            left_fit = np.polyfit(lefty, leftx, 2)
            left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                                    /np.absolute(2*left_fit[0])
            left_fit_cr = np.polyfit(lefty , leftx , 2)
            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * left_fit_cr[0])

        left_fitx = sanity_check(left_lane, left_fit, yvals,  left_curverad)


        if len(righty)>5:
            y_eval = np.max(righty)
            right_fit = np.polyfit(righty, rightx, 2)
            right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                        /np.absolute(2*right_fit[0])
            right_fit_cr = np.polyfit(righty , rightx , 2)
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * right_fit_cr[0])

        right_fitx = sanity_check(right_lane, right_fit, yvals, right_curverad)

        return left_fitx, right_fitx, yvals

    def sanity_check(lane, polyfit, yvals, curvature):
        if  len(lane.polyfit) <= 5:
            lane.radius_of_curvature = curvature
            lane.polyfit = polyfit
            lane.count_skip = 0
        else:
            a = np.column_stack((lane.polyfit[0] * yvals ** 2 + lane.polyfit[1] * yvals + lane.polyfit[2], yvals))
            b = np.column_stack((polyfit[0] * yvals ** 2 + polyfit[1] * yvals + polyfit[2], yvals))
            ret = cv2.matchShapes(a, b, 1, 0.0)

            if True | (ret < 0.005) | (lane.count_skip > 10):
                lane.radius_of_curvature = curvature
                lane.polyfit = polyfit
                lane.count_skip = 0
            else:
                lane.count_skip += 1

        return lane.polyfit[0] * yvals ** 2 + lane.polyfit[1] * yvals + lane.polyfit[2]


    def fillPoly(undist, warped, left_fitx, right_fitx, yvals):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = cv2.getPerspectiveTransform(warp_dst, warp_src)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = cv2.getPerspectiveTransform(warp_dst, warp_src)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        pts = np.argwhere(warped[:, :])
        position = w/2

        #pts[:,1] [pts[:,1] >= position + 40 ] = position + 40

        left  = np.mean(pts[(pts[:,1] < position) & (pts[:,0] > 410)][:,1])
        right = np.mean(pts[(pts[:,1] > position) & (pts[:,0] > 410)][:,1])

        #center = (left + right)/2
        #position = position - center

        position = -(right-w/2-40)-10

        return position

    def process_image(src_sem, src_img):
        if h == None: h = src_img.shape[0]
        if w == None: w = src_img.shape[1]
        if  len(warp_src) == 0: warp_src, warp_dst = define_warper()

        img = np.copy(src_sem)
        img = warper(img)
        img = sem2bin(img)

        leftx, lefty, rightx, righty = peaks_histogram(img)
        left_fitx, right_fitx, yvals = curvature(leftx, lefty, rightx, righty)
        position = fillPoly(src_img, img, left_fitx, right_fitx, yvals)

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

        main()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')