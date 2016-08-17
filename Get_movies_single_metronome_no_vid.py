# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 01:09:37 2016

@author: pawan
"""


import cv2

import numpy as np
import csv 
import scipy.spatial.distance as scidist
import matplotlib.pyplot as plt 
import string


#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#writer = cv2.VideoWriter('test_condition_1.avi',fourcc, 50, (1280,720))


#fileLoc2 = '/media/pawan/0B6F079E0B6F079E/Postdoc stuff/Metronome experiment/Videos/1st august 2016/DSC_0004.mov'
#fileLoc2 = 'E:/Postdoc stuff/Metronome experiment/Videos/1st august 2016/DSC_0004.mov'

#print(os.listdir())
#fileLoc2 = '/media/pawan/0B6F079E0B6F079E/Postdoc stuff/Metronome experiment/Videos/August 3 rd 2016/3_mets_208bpm_SO_trial2.MOV'
#fileLoc2 = '/media/pawan/0B6F079E0B6F079E/Postdoc stuff/Metronome experiment/Videos/shr-ram single metronome-calibration/DSC_0040.MOV'
fileLoc2 = '/media/pawan/0B6F079E0B6F079E/Postdoc stuff/Metronome experiment/Videos/shr-ram single metronome-calibration/DSC_0040.MOV'

camera = cv2.VideoCapture(fileLoc2)
framenum = 0 
camera.set(1,600)

total_num_frames = camera.get(7)

# for yellow color we have the following lower and upper rgb values 
#circleLower = np.array([70 ,  100 , 0 ],dtype=np.uint8)
#circleUpper = np.array([190,  190 , 100],dtype=np.uint8)
#circleLower = np.array([70 ,  80 , 0 ],dtype=np.uint8)
#circleUpper = np.array([190,  150 , 60],dtype=np.uint8)

circleLower = np.array([80 , 110 , 0 ],dtype=np.uint8)
circleUpper = np.array([190,  190 ,80],dtype=np.uint8)

#
#boxLower = np.array([150 ,50,  100 ],dtype=np.uint8)
#boxUpper = np.array([250, 120 , 180],dtype=np.uint8)
#boxLower = np.array([100 ,30,  60 ],dtype=np.uint8)
#boxUpper = np.array([150, 100 , 180],dtype=np.uint8)


boxLower = np.array([140 ,50,  20 ],dtype=np.uint8)
boxUpper = np.array([190, 100 , 70],dtype=np.uint8)


#record data 
rect_position = []
circle_position = []
rectangle_orientation_all =[]
circle_orientation_all = []
#data_file = csv.writer('data_file.csv'); 
# grab() is method of the class VideoCapture
# it grabs the next frame from video file or camera and return true (non-zero)
# in the case of success.

print(camera.grab())

#then you open the camera and start analysis the images 

# the while loop goes over the full video, keeps grabbing frames until the 
# return value of grabbed is false where it means that the video has ended 
 
 #and writer.isOpened()
skipped_frame_list = []
no_detection_list =[] 
framenum = 0
#with open('position_data_single_mets.csv', 'w') as csvfile:
#   csvwrite = csv.writer(csvfile,delimiter = ',', quoting=csv.QUOTE_NONE,escapechar=' ')
while(camera.isOpened() ):
  frame= 0 
  mask2= 0 
  mask2 = 0
  
  (grabbed, frame) = camera.read()
  framenum = framenum+1

  if not grabbed: 
   break 
  
  colour_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   # creating a mask by applying the greenLower and greenUpper, the mask is a binary 
  # image(doing np.unique(mask) yields (0,255)) 
  mask = cv2.inRange(colour_frame,boxLower,boxUpper)
  mask2 = cv2.inRange(colour_frame,circleLower,circleUpper)
  
  #erosion and dilation, fairly standard operations, the None refers to the 
  # the lack of a kernel. For more on morphological operations, refer to 
  #http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
  
  mask = cv2.erode(mask, None, iterations=2)
  mask = cv2.dilate(mask, None, iterations=2)
  mask2 = cv2.erode(mask2, None, iterations=2)
  mask2 = cv2.dilate(mask2, None, iterations=2)
 
#  cv2.startWindowThread()
#  cv2.imshow('duck',mask)
#  plt.imshow(colour_frame)
 ## FIND CONTOURS- this part of the code gives us rect_contours and circle_contours 
 
  cnts = 0
  cnts2 = 0
  
  (pink_image,rect_contours,hierarchy) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
  (circle_image,circle_contours,hierarchy) = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
  
  
  
  
  circle_moments = cv2.moments(circle_contours[0])   
  rect_moments = cv2.moments(rect_contours[0])   
  
  
  c_centroid_x = int(circle_moments['m10']/circle_moments['m00'])
  c_centroid_y = int(circle_moments['m01']/circle_moments['m00'])
     
  rect_centroid_x = int(rect_moments['m10']/rect_moments['m00'])
  rect_centroid_y = int(rect_moments['m01']/rect_moments['m00'])
          
  cv2.namedWindow('duck',cv2.WINDOW_NORMAL)
#          
  cv2.imshow('duck',colour_frame)
          
  if cv2.waitKey(1) & 0xFF == ord('q'):
             break
         
         
camera.release()          
          
                  