#!/usr/bin/env python3
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import imutils
from scipy.ndimage.filters import gaussian_filter
# from kalman import KalmanFilter
from imutils.video import VideoStream
import numpy.linalg as la

# Create pipeline
pipeline = dai.Pipeline()
# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("video")
# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setVideoSize(1920, 1080)
camRgb.setVideoSize(1200, 800)
# camRgb.setVideoSize(640, 400)
camRgb.setFps(60)
xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)
# Linking
camRgb.video.link(xoutVideo.input)

def detect(frame):
    # print(frame)
    orangeLower = (6, 150, 200)
    orangeUpper = (25, 255, 255)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # cv2.imshow("orange", mask)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10 and center != None:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            return center

def kalman(x_esti,P,A,Q,B,u,z,H,R):

    x_pred = A @ x_esti + B @ u;         # B : controlMatrix -->  B @ u : gravity
    #  x_pred = A @ x_esti or  A @ x_esti - B @ u : upto
    P_pred  = A @ P @ A.T + Q;

    zp = H @ x_pred

    # si no hay observación solo hacemos predicción 
    if z is None:
        return x_pred, P_pred, zp

    epsilon = z - zp

    k = P_pred @ H.T @ la.inv(H @ P_pred @ H.T +R)

    x_esti = x_pred + k @ epsilon;
    P  = (np.eye(len(P))-k @ H) @ P_pred;
    return x_esti, P, zp


###################### Kalman Initialization ########################

fps = 60
dt = 1/fps
# t = np.arange(0,2.01,dt)
noise = 3

# A : transitionMatrix
A = np.array(
    [1, 0, dt, 0,
    0, 1, 0, dt,
    0, 0, 1, 0,
    0, 0, 0, 1 ]).reshape(4,4)

# Adjust A to fit vertical velocity
a = np.array([0, 195])
# B : controlMatrix
B = np.array(
    [dt**2/2, 0,
    0, dt**2/2,
    dt, 0,
    0, dt ]).reshape(4,2)
# H : measurementMatrix
H = np.array(
    [1,0,0,0,
    0,1,0,0]).reshape(2,4)

# x, y, vx, vy
mu = np.array([0,0,0,0])
P = np.diag([1000,1000,1000,1000])**2
res=[]

sigmaM = 0.0001
sigmaZ = 3*noise

Q = sigmaM**2 * np.eye(4)   # processNoiseCov
R = sigmaZ**2 * np.eye(2)   # measurementNoiseCov
listCenterX=[]
listCenterY=[]
listpuntos=[]

add_count = 0

################# Connect to device and start pipeline #################
# Output video
width = 1200
height = 800
myvideo=cv2.VideoWriter("/out/vidout.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))

with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    startTime = time.monotonic()
    counter = 0
    fps = 0
    frame = None
    while True:
        color = (0, 0, 255)
        videoIn = video.get()
        frame = videoIn.getCvFrame()
        # print(frame.shape[0], frame.shape[1])
        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color)
        if cv2.waitKey(1) == ord('q'):
            break
        xo, yo = None, None
        if (frame is None):
            cv2.imshow('Frame', frame)
            continue
        # print(frame)
        coords = detect(frame)
        # print(coords)
        if coords == None:
            cv2.imshow('Frame', frame)
        else:
            xo, yo = coords
            cv2.imshow("Frame", frame)
        
            mu,P,pred= kalman(mu,P,A,Q,B,a,np.array([xo,yo]),H,R)
            listCenterX.append(xo)
            listCenterY.append(yo)

            res += [(mu,P)]

            ##### Prediction #####
            mu2 = mu
            P2 = P
            res2 = []

            for _ in range(120*2):
                mu2,P2,pred2= kalman(mu2,P2,A,Q,B,a,None,H,R)
                res2 += [(mu2,P2)]
                
            xe = [mu[0] for mu,_ in res]
            xu = [2*np.sqrt(P[0,0]) for _,P in res]
            ye = [mu[1] for mu,_ in res]
            yu = [2*np.sqrt(P[1,1]) for _,P in res]
            
            xp=[mu2[0] for mu2,_ in res2]
            yp=[mu2[1] for mu2,_ in res2]

            xpu = [2*np.sqrt(P[0,0]) for _,P in res2]
            ypu = [2*np.sqrt(P[1,1]) for _,P in res2]

            for n in range(len(listCenterX)): # Track centre of ball
                cv2.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)

            for n in [-1]:#range(len(xe)): # Estimate ball
                incertidumbre=(xu[n]+yu[n])/2
                cv2.circle(frame,(int(xe[n]),int(ye[n])),int(incertidumbre),(255, 255, 0),1)

            for n in range(len(xp)): # estimate trajectory
                incertidumbreP=(xpu[n]+ypu[n])/2
                cv2.circle(frame,(int(xp[n]),int(yp[n])),1,(0, 0, 255))

        myvideo.write(frame)
