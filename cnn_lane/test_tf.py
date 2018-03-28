import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

import tensorflow as tf

import numpy as np
import glob
import cv2
import math
from matplotlib import pyplot as plt

h, w = None, None

image_filename_format = '/home/kvasnyj/temp/images/{:s}/image_{:0>5d}.png'

poly_min = [0.0, 0.0, 197.94484600000001, 0.0, 0.0, 55.296128000000003]
poly_range = [0.001227, 1.1010660000000001, 411.36781700000006, 0.0082349999999999993, 8.6811699999999998, 2673.026965]

def fillPoly(undist, polyfit, scr_poly):
    yvals = np.arange(0, 390, 1.0)
    
    left_fitx = polyfit[0] * yvals ** 2 + polyfit[1] * yvals + polyfit[2]
    right_fitx = polyfit[3] * yvals ** 2 + polyfit[4] * yvals + polyfit[5]
    color_warp = np.zeros((h, w, 3), dtype='uint8')
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    #cv2.polylines(color_warp, np.int_([pts_left]), 1, (255, 0, 0)) 
    #cv2.polylines(color_warp, np.int_([pts_right]), 1, (255, 0, 0)) 
    res = cv2.addWeighted(undist, 1, color_warp, 0.3, 0)  

    left_fitx_src = scr_poly[0] * yvals ** 2 + scr_poly[1] * yvals + scr_poly[2]
    right_fitx_src = scr_poly[3] * yvals ** 2 + scr_poly[4] * yvals + scr_poly[5]        
    color_warp_src = np.zeros((h, w, 3), dtype='uint8')
    pts_left_src = np.array([np.transpose(np.vstack([left_fitx_src, yvals]))])
    pts_right_src = np.array([np.flipud(np.transpose(np.vstack([right_fitx_src, yvals])))])
    pts_src = np.hstack((pts_left_src, pts_right_src))
    cv2.fillPoly(color_warp_src, np.int_([pts_src]), (0, 0, 255))    
    res = cv2.addWeighted(res, 1, color_warp_src, 0.3, 0)  

    return res



def cnn_lanes(img, frame, scr_poly):
    global h, w
    if h == None: h = img.shape[0]
    if w == None: w = img.shape[1]

    x = image_pipeline(img)

    feed_dict = {X:x}

    poly = sess.run(model, feed_dict)
    poly = poly.reshape(6)
    
    polyfit = poly * poly_range + poly_min

    img = fillPoly(img, polyfit, scr_poly)

    cv2.imwrite(image_filename_format.format('cnn', frame), img)

def image_pipeline(img):
    img = img[:370, 360:, :]

    h,w,c = img.shape
    img = cv2.resize(img, (int(h/5), int(w/5)))

    #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #plt.imshow(img)
    #plt.pause(0.001)

    data = np.asarray(img, dtype="float")
    data = data / 255. # normalization

    data = data[np.newaxis, :, :]
    return data

sess = tf.Session()
saver = tf.train.import_meta_graph('./tf_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")

model = graph.get_tensor_by_name("output/BiasAdd:0") # graph.as_graph_def().node

txt = np.loadtxt("/home/kvasnyj/Dropbox/carla/cnn_lane/data/data.txt", delimiter=";")

for t in txt:
    frame = int(t[0])
    if frame>300: break

    file = "/home/kvasnyj/Dropbox/carla/cnn_lane/data/image_{:0>5d}.png".format(frame)
    img = cv2.imread(file)
    cnn_lanes(img, frame, t[1:])

print("Done")