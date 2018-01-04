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

show_camera = True
save_to_disk = True

Kp = 0.003
Ki = 0
Kd = 0 #3.5

prev_cte = 0
int_cte = 0
frame = 0
err = 0


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


def warper(img):
    size = img.shape

    basex = 520
    x = size[1]
    y = size[0]
    width = 100
    height = 100

    src = np.float32([
        [0, basex],
        [790, basex],
        [215, 400],
        [580, 400]
    ])

    dst = np.float32([
        [(x-width)/2, basex],
        [x - (x-width)/2, basex],
        [(x-width)/2, basex-height],
        [x - (x-width)/2, basex-height]
    ]) 

    # Compute and apply perpective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def show_and_save(sensor_data):
    image_filename_format = '/home/kvasnyj/temp/images/{:s}/image_{:0>5d}.png'

    if show_camera:
        img_sem = sensor_data.get('CameraSemanticSegmentation').data
        img = np.copy(sensor_data.get('CameraRGB').data)
        img_depth = np.copy(sensor_data.get('CameraDepth').data)
        img_depth = img_depth*255

        img_sem = warper(img_sem)
        img_size = img.shape

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

            show_and_save(sensor_data)

            sem = sensor_data.get('CameraSemanticSegmentation').data
            depth = sensor_data.get('CameraDepth').data
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
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
