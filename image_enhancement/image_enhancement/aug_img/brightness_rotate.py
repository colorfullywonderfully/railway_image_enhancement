#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:14:11 2019

@author: wu
"""

import cv2
import os

import tensorflow as tf

img_path = "./img/"
save_path= "./out/"
count = 0
a = 50

for root, dir, files in os.walk(img_path):
    for file in files:
        srcImg = cv2.imread(img_path + "/" + str(file), 0)    
        result = srcImg.copy()
       
        RotateMatrix = cv2.getRotationMatrix2D(center=(result.shape[1] / 2, result.shape[0] / 2), angle=4, scale=1)
        rotImg = cv2.warpAffine(result, RotateMatrix, (result.shape[1], result.shape[0]))

        img_gray = rotImg.copy()
        rows, cols = img_gray.shape
        

 
        src_RGB = cv2.cvtColor(img_gray , cv2.COLOR_GRAY2BGR)
        #pic = cv2.resize(src_RGB, (800, 600), interpolation=cv2.INTER_CUBIC)

        #  random_brightness = tf.image.random_brightness(pic ,max_delta=30)
        
        random_contrast = tf.image.random_contrast(src_RGB,lower=1,upper=3)
        new_img = tf.image.adjust_brightness(random_contrast, 0.18)
        sess = tf.Session()
        #sess.run(tf.global_variables_initializer())
#        #                               ##转化为numpy数组  
        # src_RGB = exposure.adjust_gamma(src_RGB, 0.8)
        img_numpy = new_img.eval(session=sess)

        cv2.imwrite(save_path + str(file),  img_numpy , [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        count += 1

print(count)
