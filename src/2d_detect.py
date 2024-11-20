#!/usr/bin/env python3
from pathlib import Path
import cv2
import depthai as dai
# import numpy as np
import time
import argparse
import imutils
# from scipy.ndimage.filters import gaussian_filter
# from kalman import KalmanFilter
from imutils.video import VideoStream

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

# Connect to device and start pipeline
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
