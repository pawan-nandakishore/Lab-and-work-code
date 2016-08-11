# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:37:26 2016

@author: pawan
"""

import cv2

import numpy as np
import csv 
import scipy.spatial.distance as scidist
import matplotlib.pyplot as plt 
import string


fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('test_condition_1.avi',fourcc, 50, (1280,720))


#fileLoc2 = '/media/pawan/0B6F079E0B6F079E/Postdoc stuff/Metronome experiment/Videos/1st august 2016/DSC_0004.mov'
#fileLoc2 = 'E:/Postdoc stuff/Metronome experiment/Videos/1st august 2016/DSC_0004.mov'

#print(os.listdir())
fileLoc2 = '/media/pawan/0B6F079E0B6F079E/Postdoc stuff/Metronome experiment/Videos/August 3 rd 2016/3-mets-208bpm-SO-trial3-set2.MOV'
camera = cv2.VideoCapture(fileLoc2)
framenum = 0 
camera.set(1,5)

total_num_frames = camera.get(7)

# for yellow color we have the following lower and upper rgb values 
#circleLower = np.array([70 ,  100 , 0 ],dtype=np.uint8)
#circleUpper = np.array([190,  190 , 100],dtype=np.uint8)
circleLower = np.array([70 ,  80 , 0 ],dtype=np.uint8)
circleUpper = np.array([190,  150 , 60],dtype=np.uint8)
#
#boxLower = np.array([150 ,50,  100 ],dtype=np.uint8)
#boxUpper = np.array([250, 120 , 180],dtype=np.uint8)
boxLower = np.array([100 ,30,  60 ],dtype=np.uint8)
boxUpper = np.array([150, 100 , 180],dtype=np.uint8)



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
with open('position_data_multiple_mets.csv', 'w') as csvfile:
        csvwrite = csv.writer(csvfile,delimiter = ',', quoting=csv.QUOTE_NONE,escapechar=' ')
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
          #cv2.StartWindowThread()
#          cv2.startWindowThread()
#          cv2.imshow('duck',mask)
#          plt.imshow(colour_frame)
         ## FIND CONTOURS- this part of the code gives us rect_contours and circle_contours 
         
          cnts = 0
          cnts2 = 0
          
          (pink_image,rect_contours,hierarchy) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        		cv2.CHAIN_APPROX_SIMPLE)
          (circle_image,circle_contours,hierarchy) = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
        		cv2.CHAIN_APPROX_SIMPLE)
          
          
          ## DETECT CONTOUR SHAPE AND SIZE - we detect how many contours have four verticies and also 
           # find the countour area. Based on this we assign the number of number of metronomes in the 
          # image 
          num_verticies = [] 
          for curve in rect_contours :   
             perimeter = cv2.arcLength(curve, True)
             curve_approximation = cv2.approxPolyDP(curve, 0.04 * perimeter, True)
             num_verticies.append(len(curve_approximation)) 
             
          # rectangle_indices tells us which contours in rect_contours are rectangles. If it is a structure that has 
          # 4 verticese then it is a rectangular region   
          rectangle_indices= [] 
          rectangle_indices = np.where(np.asarray(num_verticies)==4)
          
          
          # we now check for area of the rectangular regions 
          rect_contourAreas = []  
          for indx in rectangle_indices[0] :   
            rect_contourAreas.append(cv2.contourArea(rect_contours[indx]))
            
          # based on the area we select the final set of indices whose contours for a rectangle   
          final_rectangle_indices = []
          final_rectangle_indices =  np.where((np.asarray(rect_contourAreas)>5000) & (np.asarray(rect_contourAreas)<7000))
        
          temp_rect_contours = [] 
          for contour in final_rectangle_indices[0] :
              temp_rect_contours.append(rect_contours[contour])
          
          rect_contours = temp_rect_contours
          
          #cv2.namedWindow('duck',cv2.WINDOW_NORMAL)
          
          #cv2.imshow('duck',mask2)
          
          #plt.imshow(colour_frame)
          
          
          
          
          ## SELECT CIRCULAR CONTOURS- based on the number of rectangular contours we select the largest of the circular contours
          # for this first we calcuate the area of all the circular contours 
          
          circ_contour_areas = []
          for circ in circle_contours : circ_contour_areas.append(cv2.contourArea(circ))  
          
          sorted_indices = np.argsort(np.asarray(circ_contour_areas),axis= -1 )[::-1]
        
          final_circ_indx = sorted_indices[0:len(final_rectangle_indices[0])]
          
          
         # final circular contours is found and then assigned to the variable circle contours 
          final_contours = []    
          for indx in final_circ_indx: final_contours.append(circle_contours[indx]) 
        
          circle_contours = final_contours 
        
        
         # If nothing is detected then crash the program 
          if ((len(circle_contours) or  len(rect_contours)) == 0  ): 
              print('no detection :', str(framenum) ) 
              no_detection_list.append(framenum) 
              continue
             
          rect_position = [] 
          circle_position=[]
          
          
          if len(circle_contours) <3: 
              skipped_frame_list.append(framenum) 
              continue 
              
          for single_met in range(0,len(rect_contours)) :
             
          ## Moments of both the contours using which you can calculate the centroids ##
             
                  circle_moments = cv2.moments(circle_contours[single_met])   
                  rect_moments = cv2.moments(rect_contours[single_met])   
                  
                  c_centroid_x = int(circle_moments['m10']/circle_moments['m00'])
                  c_centroid_y = int(circle_moments['m01']/circle_moments['m00'])
                     
                  rect_centroid_x = int(rect_moments['m10']/rect_moments['m00'])
                  rect_centroid_y = int(rect_moments['m01']/rect_moments['m00'])
                     
                  ## calculate orientations 
                  
                  #circle orientation  
                  muc_02 =  circle_moments['m02']/circle_moments['m00']
                  muc_20 =  circle_moments['m20']/circle_moments['m00']
                  muc_11 =  circle_moments['m11']/circle_moments['m00']
                  circle_orientation = 0.5*(np.arctan( (2*muc_11)/(muc_20-muc_02)))  
                  
                  #rectangle   
                  mu_02 =  rect_moments['m02']/rect_moments['m00']
                  mu_20 =  rect_moments['m20']/rect_moments['m00']
                  mu_11 =  rect_moments['m11']/rect_moments['m00']
                  rect_orientation = 0.5*(np.arctan( (2*mu_11)/(mu_20-mu_02)))  
                  
                  
                 
                  # find the centroid positions for both the rectangle and the circle contours 
                  
                  rect_position.append((rect_centroid_x,rect_centroid_y))
                  circle_position.append((c_centroid_x,c_centroid_y))
                  
                  #rectangle_orientation_all.append(rect_orientation)
                  #circle_orientation_all.append(circle_orientation)
                  
                  
          distance_matrix = scidist.cdist(np.asarray(circle_position),np.asarray(rect_position))        
          all_positions = []
          for single_met in range(0,len(rect_contours)) :
                    
                  single_rect_contour = rect_contours[single_met]
                  single_circle_contour = circle_contours[np.argmin(distance_matrix[:,single_met])]
                  
                  c_centroid =  circle_position[np.argmin(distance_matrix[:,single_met])]     
                  c_centroid_x = c_centroid[0]
                  c_centroid_y = c_centroid[1]
                  
                  
                  rect_centroid =  rect_position[single_met]     
                  rect_centroid_x = rect_centroid[0]
                  rect_centroid_y = rect_centroid[1]
                  
                  ## plot rectangles around the detected blobs ##
                  (x,y,w,h) = cv2.boundingRect(single_rect_contour)
                  (xc,yc,wc,hc) = cv2.boundingRect(single_circle_contour)
                  sw = 110
                  sh  = 15
#                  text1 = 'x ='+ str(c_centroid_x) +' '+'y ='+ str(c_centroid_y)  
#                  text2 = 'x ='+ str(rect_centroid_x) +' '+'y ='+ str(rect_centroid_y)  
#                  text3 = 'frame number : ' + str(framenum)
#                  cv2.rectangle(colour_frame, (x, y), (x + w, y + h),(0,0,255),2 )
#                  cv2.rectangle(colour_frame, (x, y), (x+sw, y-sh),(0,0,255),-2 ,)
#                  cv2.rectangle(colour_frame, (xc, yc), (xc + wc, yc + hc),(0,0,255),2 )
#                  cv2.rectangle(colour_frame, (xc, yc), (xc+sw, yc-sh),(0,0,255),-2 ,)
#                  cv2.putText(colour_frame, text1, (x,y-5), cv2.FONT_HERSHEY_TRIPLEX, 0.4,(0,255,0)) 
#                  cv2.putText(colour_frame, text2, (xc,yc-5), cv2.FONT_HERSHEY_TRIPLEX, 0.4,(0,255,0)) 
#                  
#                  cv2.putText(colour_frame, text3, (20,40), cv2.FONT_HERSHEY_TRIPLEX, 0.7,(0,255,0)) 
                  all_positions.append((framenum,single_met,c_centroid_x,c_centroid_y) )
                  csvwrite.writerow((framenum,single_met,c_centroid_x,c_centroid_y,x,y,w,h,xc,yc,wc,hc))
                 
                  
                  
                 
                  
                         
          print(framenum)   
#          writer.write(colour_frame)  
#          cv2.namedWindow('duck',cv2.WINDOW_NORMAL)
#          
#          cv2.imshow('duck',colour_frame)
#                           
          if cv2.waitKey(1) & 0xFF == ord('q'):
             break
          
          
          
        camera.release() 
       # writer.release()

with open('skipped_frame_list.csv', 'w') as csvfile:
        csvwrite = csv.writer(csvfile,delimiter = ',', quoting=csv.QUOTE_NONE,escapechar=' ')
        csvwrite.writerow(skipped_frame_list)


with open('no_detection_list.csv', 'w') as csvfile:
        csvwrite = csv.writer(csvfile,delimiter = ',', quoting=csv.QUOTE_NONE,escapechar=' ')
        csvwrite.writerow(no_detection_list)


#with open('position_data.csv', 'w') as csvfile:
#   csvwrite = csv.writer(csvfile,delimiter = ' ', quoting=csv.QUOTE_NONE,escapechar=' ')
#   csvwrite.writerow(['Metronome poisiton'+","+' ' +","+'Bob position']) 
#   for x,y in zip(rect_position,circle_position) :
#       csvwrite.writerow([str(x[0])+","+str(x[1])+","+str(y[0])+","+str(y[1])])       
#     
#       
#with open('orientation_data.csv', 'w') as csvfile:
#   csvwrite = csv.writer(csvfile,delimiter = ' ', quoting=csv.QUOTE_NONE)
#   for x in rectangle_orientation_all :
#       csvwrite.writerow([str(x)])
#       
#       