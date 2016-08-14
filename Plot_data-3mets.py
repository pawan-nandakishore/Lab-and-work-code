# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:02:07 2016

@author: pawan
"""

import numpy as np
import csv 
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt 
import string

#with open('eggs.csv', 'rb') as csvfile:
#...     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#...     for row in spamreader:
#...         print ', '.join(row)

fileLoc='/media/pawan/0B6F079E0B6F079E/PYTHON_SCRIPTS/Shashi_detection/Metronome_error_detection/Test_condition_1/Data files/August3rd_trial2 /position_data_multiple_mets.csv' 

# OPEN FILE AND THEN READ EACH ROW. THE LIST ALL_DATA WILL BE A LIST OF LISTS WHERE ALL THE ROW DATA WILL BE APPENDED 
all_data = []
with open(fileLoc)  as csvfile: 
    
     readfile = csv.reader(csvfile, delimiter = ',')
     for row in readfile: 
         all_data.append(map(float,row[:14]))
         

all_data = np.array(all_data)         
coordinates = all_data[:,2:6]
circles=coordinates[:,0:2]
boxes = coordinates[:,2:4]


all_distances =[]
dist = DistanceMetric.get_metric('euclidean')

for x in range(0,len(all_data)): 
    dist_mat = dist.pairwise([boxes[x],circles[x]])
    all_distances.append([all_data[x,0],all_data[x,1], dist_mat[0][1]])

all_distances =np.array(all_distances)

number_mets = np.unique(all_data[:,1])

len_met_pos_list = len(all_distances)/len(number_mets)



## SORT THE FULL VALUES SUCH THAT YOU GET DATA FOR EACH OF THE METRONOMES
met_data_boxes = boxes[0:-1:3,:]
for x in range(1,len(number_mets)-1):
 met_data_boxes= np.hstack((met_data_boxes,boxes[x:-1:3,:]))

copy_boxes = np.vstack((boxes[2:-1:3,:], boxes[-1,:]))
met_data_boxes= np.hstack((met_data_boxes,copy_boxes))


met_data_circles = circles[0:-1:3,:]
for x in range(1,len(number_mets)-1):
 met_data_circles= np.hstack((met_data_circles,circles[x:-1:3,:]))

 
copy_circles = np.vstack((circles[2:-1:3,:], circles[-1,:]))
met_data_circles= np.hstack((met_data_circles,copy_circles))



met_data_dists = all_distances[0:-1:3,:]
for x in range(1,len(number_mets)-1):
 met_data_dists= np.hstack((met_data_dists,all_distances[x:-1:3,:]))
 
 
copy_distances = np.vstack((all_distances[2:-1:3,:], all_distances[-1,:]))
met_data_dists= np.hstack((met_data_dists,copy_distances))


bob_direction = met_data_circles[:,0:-1:2]- met_data_boxes[:,0:-1:2]
 
bob_direction[bob_direction<0] =-1
bob_direction[bob_direction>=0] =1

just_distances =  np.hstack((met_data_dists[:,2:-1:3],met_data_dists[:,-1:]))

dist_with_direction = np.multiply(just_distances,bob_direction)

dist_with_direction_copy = dist_with_direction.copy()

dist_with_direction[dist_with_direction<0] = np.subtract(dist_with_direction[dist_with_direction<0],-67) 

dist_with_direction[dist_with_direction>0]= np.subtract(dist_with_direction[dist_with_direction>0],67) 

