import numpy as np
import cv2
from matplotlib import pyplot as plt

def define_warper(img):
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

    return src, dst

def warper(img):
    # Compute and apply perpective transform
    M = cv2.getPerspectiveTransform(warp_src, warp_dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


file = "/home/kvasnyj/temp/images/plot/image_00023.png"
img = cv2.imread(file)

warp_src, warp_dst = define_warper(img)
warped = warper(img)

cv2.imwrite("/home/kvasnyj/temp/images/plot/image_00023-warped.png", warped)

plt.imshow(warped)

plt.pause(0)