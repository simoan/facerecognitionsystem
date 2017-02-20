# -*- coding: utf-8 -*-
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
start = time.time()

import argparse
import cv2

import itertools
import os
import numpy as np
import openface






 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

 
# allow the camera to warmup
time.sleep(0.1)

face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/haarcascade_frontalface_alt.xml')
align = openface.AlignDlib("/home/pi/openface/models/dlib/shape_predictor_68_face_landmarks.dat")
net = openface.TorchNeuralNet("/home/pi/openface/models/openface/nn4.v1.ascii.t7", 96)


def getRep(bb, rgbImg):

     start = time.time()
     alignedFace = align.align(96, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

     start = time.time()
     rep = net.forward(alignedFace)
     return rep

 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
        bgrImg = frame.array
 
	gray = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2GRAY)
	faces = None
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        
        if len(faces) != 0:
            for (x,y,w,h) in faces:
                  cv2.rectangle(bgrImg,(x,y),(x+w,y+h),(255,0,0),2)

            rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
            
            print("face detected - please wait for 1min")                
            bb = align.getLargestFaceBoundingBox(rgbImg)
            
            # LCD: face detected- bitte warten und gesicht für 1 minute stillhalte
                    
            
            rep = getRep(bb, rgbImg)
            print(rep)


        # show the frame
	cv2.imshow("Frame", bgrImg)
	key = cv2.waitKey(1) & 0xFF
        
        bb = None;
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

