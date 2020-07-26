# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:24:48 2020

@author: FENGG
"""


# -*- coding:utf-8 -*-


import cv2
import numpy as np
import os.path
import copy
 
 
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
 
 
def noiseing(img):
    #img = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY)
    param = 30
    grayscale = 256
    w = img.shape[1]
    h = img.shape[0]
    newimg = np.zeros((h, w, 3), np.uint8)
    #row and col
    for x in range(0, h):
        for y in range(0, w, 2): #Avoid exceeding boundaries
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()
            z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
            z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
 
            fxy_0 = int(img[x, y, 0] + z1)
            fxy_1 = int(img[x, y, 1] + z1)
            fxy_2 = int(img[x, y, 2] + z1)
            fxy1_0 = int(img[x, y + 1, 0] + z2)
            fxy1_1 = int(img[x, y + 1, 1] + z2)
            fxy1_2 = int(img[x, y + 1, 2] + z2)
            # f(x,y)
            if fxy_0 < 0:
                fxy_val_0 = 0
            elif fxy_0 > grayscale - 1:
                fxy_val_0 = grayscale - 1
            else:
                fxy_val_0 = fxy_0
            if fxy_1 < 0:
                fxy_val_1 = 0
            elif fxy_1 > grayscale - 1:
                fxy_val_1 = grayscale - 1
            else:
                fxy_val_1 = fxy_1
            if fxy_2 < 0:
                fxy_val_2 = 0
            elif fxy_2 > grayscale - 1:
                fxy_val_2 = grayscale - 1
            else:
                fxy_val_2 = fxy_2
            # f(x,y+1)
            if fxy1_0 < 0:
                fxy1_val_0 = 0
            elif fxy1_0 > grayscale - 1:
                fxy1_val_0 = grayscale - 1
            else:
                fxy1_val_0 = fxy1_0
            if fxy1_1 < 0:
                fxy1_val_1 = 0
            elif fxy1_1 > grayscale - 1:
                fxy1_val_1 = grayscale - 1
            else:
                fxy1_val_1 = fxy1_1
            if fxy1_2 < 0:
                fxy1_val_2 = 0
            elif fxy1_2 > grayscale - 1:
                fxy1_val_2 = grayscale - 1
            else:
                fxy1_val_2 = fxy1_2
 
            newimg[x, y, 0] = fxy_val_0
            newimg[x, y, 1] = fxy_val_1
            newimg[x, y, 2] = fxy_val_2
            newimg[x, y + 1, 0] = fxy1_val_0
            newimg[x, y + 1, 1] = fxy1_val_1
            newimg[x, y + 1, 2] = fxy1_val_2
 
        #newimg = cv2.cvtColor(newimg, cv2.COLOR_GRAY2RGB)
    cv2.destroyAllWindows()
    return newimg
 
 
 
#i = 0
file_dir = "./img/"
save_dir = "./out/"


for class_name in os.listdir(file_dir):
#for index,name in enumerate(classes):
    class_path = file_dir+class_name
#    for img_name in os.listdir(class_path):
    img_path = class_path 
    image = cv2.imread(img_path)
 
        #Simple rotation 90 degrees
    rotated = rotate(image, 5)
    #cv2.imwrite(save_dir + class_name.split('/')[-1].split('.')[0] + '_ro5.jpg', rotated)
 
        #Rotate 180 degrees and add Gaussian noise
    #rotated = rotate(image, -5)
#        
    #newimg = noiseing(rotated)
        #newimg = cv2.cvtColor(newing, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_dir + class_name.split('/')[-1].split('.')[0] + '_rono.jpg',  rotated )
 