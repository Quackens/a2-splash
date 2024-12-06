#!/usr/bin/env python3

import cv2
import math
import depthai as dai
import numpy as np
stepSize = 0.02
from pathlib import Path
import time
import argparse
import imutils
# from scipy.ndimage.filters import gaussian_filter
# from kalman import KalmanFilter
from imutils.video import VideoStream

newConfig = False

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutVideo.setStreamName("video")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setOutputSize(496, 304)
stereo.setOutputKeepAspectRatio(False)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setFps(60)

downscaleColor = True
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1920, 1080)
camRgb.setFps(60)
xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Config
topLeft = dai.Point2f(0.49, 0.49)
bottomRight = dai.Point2f(0.51, 0.51)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEAN
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

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

def coordGen(xo, yo, x, y, z):
    displacement = math.sqrt((x**2) + (y**2))#x, y = spatial cam xy, relative to center
    realZ = math.sqrt((z**2) - (displacement**2)) #z = spatial cam z, not irl Z
    return xo, yo, realZ #x, y, z in real world coords

coordList = list()

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (255, 255, 255)

    print("Use WASD keys to move ROI!")
    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    startTime = time.monotonic()
    counter = 0
    fps = 0
    frame = None

    while True:
        videoIn = video.get()
        frame = videoIn.getCvFrame()
        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color)
        xo, yo = None, None
        # if (frame is None):
        #     # cv2.imshow('Frame', frame)
        #     continue
        # # print(frame)
        coords = detect(frame) #tuple (x, y) in pixels
        if coords == None:
            # cv2.imshow('Frame', frame)
            xo, yo = None, None
        else:
            # print(coords)
            # cv2.imshow("Frame", frame)
            xo, yo = coords


        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame() # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        spatialData = spatialCalcQueue.get().getSpatialLocations()
        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            # print("------------------------------new frame----------------")
            # print("x: "+str(int(depthData.spatialCoordinates.x))+"y: "+str(int(depthData.spatialCoordinates.y))+"z: "+str(int(depthData.spatialCoordinates.z)))
            if (int(depthData.spatialCoordinates.z)) != 0:
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)
                # cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
                # cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
                # cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)
                newCoord = coordGen(xo, yo, int(depthData.spatialCoordinates.x), int(depthData.spatialCoordinates.y), int(depthData.spatialCoordinates.z))
            if (coords != None):
                # print(newCoord)
                coordList.append(newCoord)
                # print(coordList[-1])
        # Show the frame
        cv2.putText(depthFrameColor, "FPS: {:.2f}".format(fps), (2, depthFrameColor.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color)
        
        #constant ball detection update
        if coords != None:
            topLeft = dai.Point2f((xo-10)/frame.shape[1], (yo-10)/frame.shape[0])
            bottomRight = dai.Point2f((xo+10)/frame.shape[1], (yo+10)/frame.shape[0])
            newConfig = True

        cv2.imshow("depth", depthFrameColor)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('1'): 
            # print(depthFrameColor.shape[0], depthFrameColor.shape[1]) = 400, 640
            print((xo-5)/depthFrameColor.shape[1], (yo-5)/depthFrameColor.shape[0])
            print(coords)
            topLeft = dai.Point2f((xo-5)/depthFrameColor.shape[1], (yo-5)/depthFrameColor.shape[0])
            bottomRight = dai.Point2f((xo+5)/depthFrameColor.shape[1], (yo+5)/depthFrameColor.shape[0])
            newConfig = True
        elif key == ord('2'): 
            print(coordList)
        elif key == ord('3'): 
            # print(depthFrameColor.shape[0], depthFrameColor.shape[1]) = 400, 640
            # print((xo-5)/depthFrameColor.shape[1], (yo-5)/depthFrameColor.shape[0])
            # print(coords)
            topLeft = dai.Point2f((0)/depthFrameColor.shape[1], (0)/depthFrameColor.shape[0])
            bottomRight = dai.Point2f((5)/depthFrameColor.shape[1], (5)/depthFrameColor.shape[0])
            newConfig = True
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True

        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)
            newConfig = False
