# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:44:40 2018

@author: COSH
"""

import argparse

import cv2
import numpy as np
import math
import os
from objloader import *

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10  
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N/2 < 4:
        print('At least 4 points should be given')
    A = np.array([[u[0],u[1],1,0,0,0,-u[0]*v[0],-u[1]*v[0],-v[0]],
                      [0,0,0,u[0],u[1],1,-u[0]*v[1],-u[1]*v[1],-v[1]],
                      [u[2],u[3],1,0,0,0,-u[2]*v[2],-u[3]*v[2],-v[2]],
                      [0,0,0,u[2],u[3],1,-u[2]*v[3],-u[3]*v[3],-v[3]],
                      [u[4],u[5],1,0,0,0,-u[4]*v[4],-u[5]*v[4],-v[4]],
                      [0,0,0,u[4],u[5],1,-u[4]*v[5],-u[5]*v[5],-v[5]],
                      [u[6],u[7],1,0,0,0,-u[6]*v[6],-u[7]*v[6],-v[6]],
                      [0,0,0,u[6],u[7],1,-u[6]*v[7],-u[7]*v[7],-v[7]]]).astype(float)
    
    a = np.dot(A.T, A)
    eigValue, eigVect= np.linalg.eig(a)
    value,index= find_nearest( eigValue, 0 )
    H = np.reshape(eigVect[:,index],(3,3))
    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    height, width, ch = img.shape
    height = height-1
    width = width-1
    u = np.array([0, 0, 0 ,  height - 1, width - 1, height -1, width-1, 0 ])
    v = np.array([corners[0,0],corners[0,1],corners[1,0],corners[1,1],
                      corners[2,0],corners[2,1],corners[3,0],corners[3,1]])
    # TODO: some magic
    H = solve_homography(u, v)
    
    height = height+1
    width = width+1
    ux,uy = np.meshgrid(np.arange(0,width,1),np.arange(0,height,1))
    a = np.ones(width*height)  ### 1
    ux = np.reshape(ux,(1,width*height)) ###ux
    uy = np.reshape(uy,(1,width*height)) ###uy
    ux = np.r_[ux,uy] 
    ux = np.c_[ux.T,a] 
    c = np.dot(np.reshape(H,(3,3)),ux.T)
    a = np.array([c[2,:],c[2,:],c[2,:]])
    b = np.floor(c[:,:]/a) 
    b = b.astype(int)
    vx = b[1,:]
    vy = b[0,:]
       
    return vx,vy
      
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx],idx    
  
def main():
    """
    This functions loads the target surface image,
    """
    homography = None 
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    model = cv2.imread('./input/IvcnWasc.png')
    input_model = cv2.imread('./input/crosswalk_top.jpg')
    h,w,chi = model.shape
    hi,wi,ch = input_model.shape
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    # init video capture
    cap = cv2.VideoCapture('./input/ar_marker.mp4')
    output = cv2.VideoWriter('./input/output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),25,(1920,1080))
    k=0
    
    while True:
        k+=1
        # read the current frame
        ret, frame = cap.read()
#        cv2.imwrite('r'+str(k)+'.png', frame)
        
        height = 960*2
        width = 540*2
        frame = cv2.resize(frame,(height,width))
        frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame2 = cv2.resize(frame2,(height,width))
        kp_frame, des_frame = orb.detectAndCompute(frame2, None)
        matches = bf.match(des_model, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        input_model = np.reshape(input_model,(hi,wi,3))
        if len(matches) > MIN_MATCHES:
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10)
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            #
            dst = cv2.perspectiveTransform(pts, homography)
            dst = np.reshape(dst,(4,2))
#            A = frame[357:835,503:1243]
#            cv2.imwrite('part4.png', A)
            print(k)
            vx,vy = transform(input_model,frame, dst)
            input_model = np.reshape(input_model,(hi*wi,3))
            frame[1080,1920,:] = input_model[:,:]
            cv2.imwrite(str(k)+'.png', frame)
            output.write(frame)
#            
    cap.release()
    output.release()
    
if __name__ == '__main__':
    main()        