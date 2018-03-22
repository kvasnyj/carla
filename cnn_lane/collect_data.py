# ~/repo/carla/CarlaUE4.sh /Game/Maps/Town01 -carla-server -fps=15 -windowed -ResX=800 -ResY=600 
from __future__ import print_function

import argparse
import logging
import random
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
import cv2

import sys
sys.path.insert(0, '/home/kvasnyj/Dropbox/carla/')
from carla.client import make_carla_client
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from Line import Line

image_filename_format = './data/image_{:0>5d}.png'
show_camera = False
save_to_disk = False

Kp = 0.008
Ki = 0.0001
Kd = 0
prev_cte = 0
int_cte = 0
frame = 0
err = 0

# image shape
h, w = None, None
warp_src, warp_dst = [], []

video = None

def UpdateError(cte):
    global prev_cte, int_cte, err
    diff_cte = cte - prev_cte
    prev_cte = cte
    int_cte += cte
    err += (1 + abs(cte)) * (1 + abs(cte))

    steer = -Kp * cte - Kd * diff_cte - Ki * int_cte
    if steer > 0.3: steer = 0.3
    if steer < -0.3: steer = -0.3

    print(cte, diff_cte, int_cte)

    return steer


def TotalError():
    return err / frame

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

    return left_fit, right_fit

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


def define_position(warped):
    pts = np.argwhere(warped[:, :])
    position = w/2

    #pts[:,1] [pts[:,1] >= position + 40 ] = position + 40

    left  = np.mean(pts[(pts[:,1] < position) & (pts[:,0] > 410)][:,1])
    right = np.mean(pts[(pts[:,1] > position) & (pts[:,0] > 410)][:,1])
    
    #center = (left + right)/2
    #position = position - center


    position = -(right-w/2-40)-10

    if np.isnan(left): left = 0
    if np.isnan(right): right = 0
    if np.isnan(position): position = 0

    print(left, right, position)

    if show_camera:
        plt.imshow(img)
        plt.pause(0.001)

    return int(left), int(right), position 

def process_image(src_sem, src_img):
    global h, w, warp_src, warp_dst
    if h == None: h = src_img.shape[0]
    if w == None: w = src_img.shape[1]
    if  len(warp_src) == 0: warp_src, warp_dst = define_warper()
        
    img = np.copy(src_sem)
    img_warped = warper(img)
    img_bin = sem2bin(img_warped)
    if save_to_disk:
        cv2.imwrite(image_filename_format.format('Bin', frame), img_bin)

    if frame>=10:
        leftx, lefty, rightx, righty = peaks_histogram(img)
        left_fit, right_fit = curvature(leftx, lefty, rightx, righty)  
        if left_fit[2]>0 and right_fit[2]>0:   
            cv2.imwrite("/home/kvasnyj/Dropbox/carla/cnn_lane/data/image_{:0>5d}.png".format(frame), src_img)
            with open("/home/kvasnyj/Dropbox/carla/cnn_lane/data/data.txt", "a") as text:
                text.write("%i;%f;%f;%f;%f;%f;%f\r\n" % (frame,left_fit[0],left_fit[1],left_fit[2],right_fit[0],right_fit[1],right_fit[2]))        


    left, right, position  = define_position(img_bin)

    front = img_warped[h-300-10:h-300, int(left*1.1):int(right*0.9)]
    front_car = 10 in front

    red = False

    red_threshold = 200
    green_threshold = 50
    blue_threshold = 50
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    binary = (src_img[0:int(h*1), int(w/2):w])[src_sem[0:int(h*1),int(w/2):w] ==12]

    thres_img = (binary[:, 0] > rgb_threshold[0]) & \
                    (binary[:,1] < rgb_threshold[1]) & \
                    (binary[:, 2] < rgb_threshold[2])

    unique, counts = np.unique(thres_img, return_counts=True)
    d = dict(zip(unique, counts))
    if True in d:
        print('red: ', d.get(True))
        if d.get(True)>10:
            red = True

    if front_car or red: 
        return float('nan')
    else:
        return position    

def show_and_save(sensor_data, steering):
    global video
    if frame<28: return

    if save_to_disk:
        img_sem = sensor_data.get('CameraSemanticSegmentation').data
        img = np.copy(sensor_data.get('CameraRGB').data)
        img_depth = np.copy(sensor_data.get('CameraDepth').data)
        img_depth = img_depth*255

        for i in range(h):
            for j in range(w):
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

        cv2.imwrite(image_filename_format.format('plot', frame), img)

    if save_to_disk:
        #sensor_data.get('CameraRGB').save_to_disk(image_filename_format.format('CameraRGB', frame))
        if video == None: 
            video = cv2.VideoWriter('/home/kvasnyj/temp/carla.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10, (w,h))
        img = np.copy(sensor_data.get('CameraRGB').data)
        cv2.putText(img, "Steering = {:10.4f}".format(steering), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        video.write(img)

def run_carla_client(host, port):
    global frame
    with make_carla_client(host, port) as client:
        print('CarlaClient connected')
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=100,
            NumberOfPedestrians=0,
            WeatherId= 1)  # random.choice([1, 3, 7, 8, 14]))
        settings.randomize_seeds()

        camera0 = Camera('CameraRGB')
        camera0.set_image_size(800, 600)
        camera0.set_position(30, 0, 130)
        settings.add_sensor(camera0)

        camera1 = Camera('CameraDepth', PostProcessing='Depth')
        camera1.set_image_size(800, 600)
        camera1.set_position(30, 0, 130)
        settings.add_sensor(camera1)

        camera2 = Camera('CameraSemanticSegmentation', PostProcessing='SemanticSegmentation')
        camera2.set_image_size(800, 600)
        camera2.set_position(30, 0, 130)
        settings.add_sensor(camera2)

        scene = client.load_settings(settings)

        number_of_player_starts = len(scene.player_start_spots)
        player_start =  random.randint(0, max(0, number_of_player_starts - 1))

        print('Starting...')
        client.start_episode(player_start)

        if show_camera:
            plt.ion()
            plt.show()

        while (frame < 10000):
            frame += 1
            print("---------------", frame)
            measurements, sensor_data = client.read_data()

            #print_measurements(measurements)
            #measurements.non_player_agents[0]

            sem = sensor_data.get('CameraSemanticSegmentation').data
            #depth = sensor_data.get('CameraDepth').data
            img = sensor_data.get('CameraRGB').data
            
            if measurements.player_measurements.forward_speed>0:
                position = process_image(sem, img)
            else:
                 position = 0

            if position != position:
                throttle = 0
                brake = True
            else:
                cte = position
                brake = False

                steer_value = UpdateError(cte)

                print(steer_value)
                show_and_save(sensor_data, steer_value)

                throttle = 1;


            if  (measurements.player_measurements.forward_speed >= 30):
                throttle = 0

            client.send_control(measurements.player_measurements.autopilot_control)

            # client.send_control(
            #     steer=steer_value,
            #     throttle=throttle,
            #     brake=brake,
            #     hand_brake=False,
            #     reverse=False)

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

        cv2.destroyAllWindows()
        if video != None: video.release()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
