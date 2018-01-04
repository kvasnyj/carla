# ~/carla/CarlaUE4.sh /Game/Maps/Town01 -carla-server -fps=15 -windowed -ResX=800 -ResY=600 

from __future__ import print_function

import argparse
import logging
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

image_filename_format = '/home/kvasnyj/temp/images/{:s}/image_{:0>5d}.png'
show_camera = False
save_to_disk = False

Kp = 0.003
Ki = 0
Kd = 0 #3.5
prev_cte = 0
int_cte = 0
frame = 0
err = 0

# image shape
h, w = None, None
warp_src, warp_dst = None, None

def UpdateError(cte):
    global prev_cte, int_cte, err
    diff_cte = cte - prev_cte
    prev_cte = cte
    int_cte += cte
    err += (1 + abs(cte)) * (1 + abs(cte))

    steer = -Kp * cte - Kd * diff_cte - Ki * int_cte
    if steer > 0.6: steer = 0.6
    if steer < -0.6: steer = -0.6

    print(cte, steer, err)

    return steer


def TotalError():
    return err / frame


def distance_to_side(sem, depth):
    img_size = sem.shape

    center = int(img_size[1] / 2)
    offroad, left, right, d_left, d_right = -1, -1, -1, -1, -1

    for i in range(img_size[0] - 1, -1, -1):
        if (sem[i][center] != 7): continue

        offroad = i

        if (right >= 0) & (left >= 0): continue  # already found

        for j in range(img_size[1]):
            if sem[i][j] == 7: continue

            if (j < center):
                if (j > left) | (left < 0): left = j
            else:
                if (j < right) | (right < 0): right = j

        if (left >= 0) & (d_left < 0): d_left = depth[i][left] * 1000
        if (right >= 0) & (d_right < 0): d_right = depth[i][right] * 1000
    return depth[offroad][center] * 1000, d_left, d_right



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

    #high_thresh, thresh_im = cv2.threshold(bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #lowThresh = 0.5 * high_thresh
    bin = cv2.Canny(bin, 0, 255)

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

        if debug:
            print(left_fitx)
            print(right_fitx)
            plt.plot(histogram)
            plt.show()

    return left_fitx, left_fity, right_fitx, right_fity

def curvature(leftx, lefty, rightx, righty, debug = False):
    leftx = np.float32(leftx)
    rightx = np.float32(rightx)
    lefty = np.float32(lefty)
    righty = np.float32(righty)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Define y-value where we want radius of curvature
    y_eval = np.max(lefty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) \
                                 /np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) \
                                    /np.absolute(2*right_fit[0])

    if debug: print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    if debug: print(left_curverad, 'm', right_curverad, 'm')

    yvals = np.arange(h - h / 2, h, 1.0)
    left_fitx = sanity_check(left_lane, left_fit, yvals,  left_curverad)
    right_fitx = sanity_check(right_lane, right_fit, yvals, right_curverad)

    return left_fitx, right_fitx, yvals, (left_curverad + right_curverad)/2

def sanity_check(lane, polyfit, yvals, curvature):
    if lane.polyfit== None: #new object
        lane.radius_of_curvature = curvature
        lane.polyfit = polyfit
        lane.detected = True
        lane.count_skip = 0
    else:
        a = np.column_stack((lane.polyfit[0] * yvals ** 2 + lane.polyfit[1] * yvals + lane.polyfit[2], yvals))
        b = np.column_stack((polyfit[0] * yvals ** 2 + polyfit[1] * yvals + polyfit[2], yvals))
        ret = cv2.matchShapes(a, b, 1, 0.0)

        if (ret < 0.005) | (lane.count_skip > 10):
            lane.radius_of_curvature = curvature
            lane.polyfit = polyfit
            lane.detected = True
            lane.count_skip = 0
        else:
            lane.detected = False
            lane.count_skip += 1

    return lane.polyfit[0] * yvals ** 2 + lane.polyfit[1] * yvals + lane.polyfit[2]


def fillPoly(undist, warped, left_fitx, right_fitx, yvals, curv):
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

    return result

def process_image(src_sem, src_img):
    global h, w, warp_src, warp_dst
    if h == None: h = src_img.shape[0]
    if w == None: w= src_img.shape[1]
    if warp_src == None: warp_src, warp_dst = define_warper()
        
    img = np.copy(src_sem)
    img = warper(img)
    img = sem2bin(img)
    
    leftx, lefty, rightx, righty = peaks_histogram(img)
    left_fitx, right_fitx, yvals, curv = curvature(leftx, lefty, rightx, righty)
    img = fillPoly(src_img, img, left_fitx, right_fitx, yvals, curv)

    if True:
        plt.imshow(img)
        plt.pause(0)

    return img    

def show_and_save(sensor_data):
    if show_camera:
        img_sem = sensor_data.get('CameraSemanticSegmentation').data
        img = np.copy(sensor_data.get('CameraRGB').data)
        img_depth = np.copy(sensor_data.get('CameraDepth').data)
        img_depth = img_depth*255

        for i in range(img_size[0]):
            for j in range(img_size[1]):
                sem = img_sem[i][j]
                if sem == 0:
                    img[i][j] = [0, 0, 0]
                elif sem == 1:
                    img[i][j] = [255, 0, 0]
                elif sem == 2:
                    img[i][j] = [0, 255, 0]
                elif sem == 3:
                    img[i][j] = [0, 0, 128]
                elif sem == 4:
                    img[i][j] = [0, 0, 255]
                elif sem == 5:
                    img[i][j] = [255, 255, 0]
                elif sem == 6:
                    img[i][j] = [0, 255, 255]
                elif sem == 7:
                    img[i][j] = [255, 0, 255]
                elif sem == 8:
                    img[i][j] = [128, 0, 0]
                elif sem == 9:
                    img[i][j] = [128, 128, 0]
                elif sem == 10:
                    img[i][j] = [0, 128, 0]
                elif sem == 11:
                    img[i][j] = [128, 0, 128]
                elif sem == 12:
                    img[i][j] = [0, 128, 128]
                else:
                    img[i][j] = [255, 255, 255]

        plt.imshow(img)
        plt.savefig(image_filename_format.format('plot', frame))
        plt.pause(0.001)

    if save_to_disk:
        # for name, image in sensor_data.items():
        sensor_data.get('CameraRGB').save_to_disk(image_filename_format.format('CameraRGB', frame))


def run_carla_client(host, port):
    global frame
    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(host, port) as client:
        print('CarlaClient connected')
        # Create a CarlaSettings object. This object is a wrapper around
        # the CarlaSettings.ini file. Here we set the configuration we
        # want for the new episode.
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=20,
            NumberOfPedestrians=40,
            WeatherId=1)  # random.choice([1, 3, 7, 8, 14]))
        settings.randomize_seeds()

        # Now we want to add a couple of cameras to the player vehicle.
        # We will collect the images produced by these cameras every
        # frame.

        # The default camera captures RGB images of the scene.
        camera0 = Camera('CameraRGB')
        # Set image resolution in pixels.
        camera0.set_image_size(800, 600)
        # Set its position relative to the car in centimeters.
        camera0.set_position(30, 0, 130)
        settings.add_sensor(camera0)

        # Let's add another camera producing ground-truth depth.
        camera1 = Camera('CameraDepth', PostProcessing='Depth')
        camera1.set_image_size(800, 600)
        camera1.set_position(30, 0, 130)
        settings.add_sensor(camera1)

        camera2 = Camera('CameraSemanticSegmentation', PostProcessing='SemanticSegmentation')
        camera2.set_image_size(800, 600)
        camera2.set_position(30, 0, 130)
        settings.add_sensor(camera2)

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for
        # the player. Here we can provide a CarlaSettings object or a
        # CarlaSettings.ini file as string.
        scene = client.load_settings(settings)

        # Choose one player start at random.
        number_of_player_starts = len(scene.player_start_spots)
        player_start = 0  # random.randint(0, max(0, number_of_player_starts - 1))

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print('Starting...')
        client.start_episode(player_start)

        if show_camera:
            plt.ion()
            plt.show()

        while (True):
            frame += 1
            # Read the data produced by the server this frame.
            measurements, sensor_data = client.read_data()

            # Print some of the measurements.
            print_measurements(measurements)



            sem = sensor_data.get('CameraSemanticSegmentation').data
            depth = sensor_data.get('CameraDepth').data
            img = sensor_data.get('CameraRGB').data
            process_image(sem, img)
            show_and_save(sensor_data)

            start_offroad, left, right = distance_to_side(sem, depth)
            print(start_offroad, left, right)

            cte = left - right

            steer_value = UpdateError(cte)

            throttle = 1;

            if  (measurements.player_measurements.forward_speed >= 30):
                throttle = 0

            # We can access the encoded data of a given image as numpy
            # array using its "data" property. For instance, to get the
            # depth value (normalized) at pixel X, Y
            #
            #     depth_array = sensor_data['CameraDepth'].data
            #     value_at_pixel = depth_array[Y, X]
            #

            client.send_control(
                steer=steer_value,
                throttle=throttle,
                brake=False,
                hand_brake=False,
                reverse=False)

def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.2f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x / 100,  # cm -> m
        pos_y=player_measurements.transform.location.y / 100,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    #print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    args = argparser.parse_args()

    log_level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)


    while True:
        try:
            run_carla_client('localhost', 2000)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
        except Exception as exception:
            logging.exception(exception)
            sys.exit(1)


if __name__ == '__main__':

    try:
        left_lane = Line()
        right_lane = Line()

        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
