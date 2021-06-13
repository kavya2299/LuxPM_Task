## these are basic system packages to access directories and files
import os
import time

## these are packages that we need to use to visualize the dataset 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

## to make data augmentations
from torchvision.transforms import ToPILImage
from PIL import Image
import imutils
import cv2

## these are basic python packages
import math
import random
import csv

path = "images.jpg"
img = mpimg.imread(path)
imgplot = plt.imshow(img)
#plt.show()

## Transform the image in the +x direction by 25%, and create an image
hor = img.shape[0]
vert = img.shape[1]
translate = np.float32([[1,0,0.25*hor],[0,1,0]])

translate_x = cv2.warpAffine(img,translate,(img.shape[1],img.shape[0]))
translate_x = plt.imshow(translate_x)
#plt.show()

## Transform the image in the +y direction by 25%, and create an image
hor = img.shape[0]
vert = img.shape[1]
translate = np.float32([[1, 0, 0], [0, 1, 0.25*vert]])

translate_y = cv2.warpAffine(img,translate,(img.shape[1],img.shape[0]))
translate_y = plt.imshow(translate_y)
#plt.show()

## Rotate the image in Z by 90 degree
angle = 90
rotate_90 = imutils.rotate(img, angle) 
rotate_90 = plt.imshow(rotate_90)
#plt.show()

## Rotate the image in Z by -90 degree
angle = -90
rotate_neg_90 = imutils.rotate(img, angle) 
rotate_neg_90 = plt.imshow(rotate_neg_90)
#plt.show()

## From the center of the image, in all directions, increase the RGB values of the pixels in a manner that, 
## each pixel from the center, the percentage drops by 1%. i.e. the center pixel's RGB will increase by 50%,
## and the next pixels in x and y directions will be 49%. This goes on and on until the increase becomes 0 %.
center_x, center_y = hor//2, vert//2
(b, g, r) = img[center_y, center_x]

img_copy = img.copy()
img_copy[center_y, center_x] = 1.5*img_copy[center_y, center_x]
imgplot = plt.imshow(img_copy)

# pixels = (right, left, up, down)
pixels = [center_x, center_y]
percent = 0.5
i = 1
while percent >= 0 and i != 50:
    percent -= 0.01
    
    img_copy[center_x - i, center_y - i : center_y + i + 1] = (1 + percent) * img_copy[center_x - i, center_y - i : center_y + i + 1]
    img_copy[center_x + i, center_y - i : center_y + i + 1] = (1 + percent) * img_copy[center_x + i, center_y - i : center_y + i + 1]
    img_copy[center_x - i : center_x + i + 1, center_y - i] = (1 + percent) * img_copy[center_x - i : center_x + i + 1, center_y - i]
    img_copy[center_x - i : center_x + i + 1, center_y + i] = (1 + percent) * img_copy[center_x - i : center_x + i + 1, center_y + i]

    # final = plt.imshow(img_copy)    
    # plt.show()
    i += 1
final = plt.imshow(img_copy)    
plt.show()