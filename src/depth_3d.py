#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
from detect import detect_frame
from threading import Thread
from queue import Queue

color = (255, 255, 255)

class Depth3D:
    def __init__(self, queue):
        self.frame_queue = Queue()
        self.coord_queue = queue # hopefully this is pass by reference

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

        downscaleColor = True
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setVideoSize(1920, 1080)
        camRgb.setFps(60)
        xoutVideo.input.setBlocking(False)
        xoutVideo.input.setQueueSize(1)

        # Config
        topLeft = dai.Point2f(0.4, 0.4)
        bottomRight = dai.Point2f(0.6, 0.6)

        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 100
        config.depthThresholds.upperThreshold = 10000
        calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
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

        self.pipeline = pipeline
        self.config = config

    def run(self):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:

            # Output queue will be used to get the depth frames from the outputs defined above
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
            spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
            Thread(target=self.process_frame, args=(self.queue,depthQueue,spatialCalcQueue,spatialCalcConfigInQueue)).start()

            video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
            startTime = time.monotonic()
            beginTime = time.monotonic()
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
                self.frame_queue.put((frame, time.time() - beginTime))

    def process_frame(self, frame, depthQueue, spatialCalcQueue, spatialCalcConfigInQueue):
        while True:
            frame, timestamp = self.frame_queue.get() # Blocking call, will wait until a new data has arrived
            xo, yo = None, None
            coords = detect_frame(frame) #tuple (x, y) in pixels
            if coords == None:
                # cv2.imshow('Frame', frame)
                continue

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
            # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            spatialData = spatialCalcQueue.get().getSpatialLocations()
            for depthData in spatialData:
                roi = depthData.config.roi
                roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                fontType = cv2.FONT_HERSHEY_TRIPLEX
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)
                cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, color)
                cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, color)
            # Show the frame
            # cv2.putText(depthFrameColor, "FPS: {:.2f}".format(fps), (2, depthFrameColor.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color)
            # cv2.imshow("depth", depthFrameColor)
            key = cv2.waitKey(1)

            
            if key == ord('q'):
                break
            

            x, y = coords
            z = int(depthData.spatialCoordinates.z)
            print(x, y, z)
            topLeft = dai.Point2f((xo-5)/frame.shape[1], (yo-5)/frame.shape[0])
            bottomRight = dai.Point2f((xo+5)/frame.shape[1], (yo+5)/frame.shape[0])

            self.config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(self.config)
            spatialCalcConfigInQueue.send(cfg)

            self.coord_queue.put((x, y, z, timestamp))