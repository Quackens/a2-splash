#!/usr/bin/env python3
from pathlib import Path
import cv2
import depthai as dai
import time

from pipeline_2d import Pipeline2D, Pipeline2D_CAM2
from queue_utils import CoordQueue2D, FrameQueue, SignalStart
from threading import Thread
from detect import detect_frame as detect_frame
from cam2 import run_cam2

import json
import sys

# TODO: Uncomment to integrate serial
import serial
import random
import math

import gcode.serial_comms_gcode as serial_comms_gcode
from improved_metrics import RealTimePlotter


def serialize_loop():

    # global s 
    global result_queue
    global result_queue_cam2
    from pipeline_2d import CUP_LEFT_X, CUP_RIGHT_X, CUP_CENTRE_X
    SIDE_LEFT_BOUND = CUP_LEFT_X
    SIDE_RIGHT_BOUND = CUP_RIGHT_X
    SIDE_CENTRE = CUP_CENTRE_X

    from cam2 import LEFT_BOUND, RIGHT_BOUND, CENTRE
    FRONT_LEFT_BOUND = LEFT_BOUND
    FRONT_RIGHT_BOUND = RIGHT_BOUND
    FRONT_CENTRE = CENTRE

    prev_x = 0
    prev_y = 0

    while True:
        
        # Coordinate update from both queues simultaneously 
        print(f"Side length: {result_queue.get_length()}, Front length: {result_queue_cam2.get_length()}")
        # print("Getting sideview..")
        coord_sideview = result_queue.get_coord()
        # print("Getting frontview..")
        # if result_queue_cam2.get_length() > 6:
        #     result_queue_cam2.reset_queue()

        coord_frontview = result_queue_cam2.get_coord()
        
        # print(f"SIDE_LEFT_BOUND: {SIDE_LEFT_BOUND}, SIDE_RIGHT_BOUND: {SIDE_RIGHT_BOUND}")
        # print(f"FRONT_LEFT_BOUND: {FRONT_LEFT_BOUND}, FRONT_RIGHT_BOUND: {FRONT_RIGHT_BOUND}")
        x, y = None, None
        if coord_sideview is None and coord_frontview is None:
            continue
        # print(f"Sideview: {coord_sideview}, Frontview: {coord_frontview}")
        if coord_sideview is not None:
            x, _ = coord_sideview
        if coord_frontview is not None:
            y, _ = coord_frontview
        # x, _ = coord_sideview
        # y, _ = coord_frontview
        if x is None: x = prev_x
        if y is None: y = prev_y

        
        # Normalize the x coordinate
        if x > SIDE_RIGHT_BOUND:
            gantry_x = 10
        elif x < SIDE_LEFT_BOUND:
            gantry_x = -10
        else:
            # gantry_x = (x - SIDE_CENTRE) / 7.5

            # TODO: fix
            gantry_x = (x - SIDE_CENTRE) / ((SIDE_RIGHT_BOUND - SIDE_LEFT_BOUND) / 2) * 10
            if gantry_x < -10: gantry_x = -10
            if gantry_x > 10: gantry_x = 10
        # Normalize the y coordinate    
        if y > FRONT_RIGHT_BOUND:
            gantry_y = -10
        elif y < FRONT_LEFT_BOUND:
            gantry_y = 10
        else:
            # TODO: fix
            gantry_y = (FRONT_CENTRE - y) / ((FRONT_RIGHT_BOUND - FRONT_LEFT_BOUND) / 2) * 10
            if gantry_y < -10: gantry_y = -10
            if gantry_y > 10: gantry_y = 10
        prev_x = x
        prev_y = y
        print(f"Send as grbl: {gantry_x} {gantry_y}")
        if grbl:
            serial_comms_gcode.gcode_goto(s, gantry_x, gantry_y)
        



def feed_frames():
    global signal_cam2
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    xoutVideo.setStreamName("video")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(1920, 1080)

    camRgb.setFps(60)

    xoutVideo.input.setBlocking(False)

    xoutVideo.input.setQueueSize(1)
    # Linking
    camRgb.video.link(xoutVideo.input)

    # Delete previous file
    # Path("coords.txt").unlink(missing_ok=True)
    
    with dai.Device(pipeline) as device:
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
            # print(f"FPS: {fps}")

            coord = detect_frame(frame)
            if coord != None and not signal_cam2.get_start():
                signal_cam2.set_start(True)
            coord_queue.put_coord(coord)

            # Writing coordinates to file
            # with open("coords.txt", "a") as f:
            #     if coord:
            #         f.write(f"[{coord[0]}, {coord[1]}],\n")
            #     else:
            #         f.write("null,\n")


if __name__ == '__main__':
    s = None
    grbl = False
    plotter = RealTimePlotter()
    plotter.show()
    
    ################# Prediction Pipeline Setup #################
    # frame_queue = FrameQueue()

    # Side camera data structures
    coord_queue = CoordQueue2D()
    result_queue = CoordQueue2D()
    debug_queue = FrameQueue()
    predict = Pipeline2D(coord_queue, result_queue, debug_queue) # side camera
    

    # Front camera data structures
    result_queue_cam2 = CoordQueue2D()
    pipeline_cam2 = Pipeline2D_CAM2()
    signal_cam2 = SignalStart()

    ################# Connect to device and start pipeline #################
    # Output video
    width = 1920
    height = 1080
    myvideo=cv2.VideoWriter("/out/vidout.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))

    # Main Thread: Start pipeline and get frames, and send to serializer

    
    if sys.argv[1] == "live":
        if len(sys.argv) > 2 and sys.argv[2] == "grbl":
            grbl = True
            # Open grbl serial port
            s = serial.Serial('/dev/tty.usbmodem21101',115200)
            # initialize grbl connection
            serial_comms_gcode.grbl_init(s)

        # Sends to gantry
        Thread(target=serialize_loop).start()

        # Takes in frame from the luxonis camera
        Thread(target=feed_frames).start()

        # Thread(target=run_cam2, args=(pipeline_cam2, result_queue_cam2, signal_cam2, )).start()
        # predict.run(s)

        # Side camera prediction pipeline
        Thread(target=predict.run, args=(s,)).start()

        # Front camera prediction pipeline
        run_cam2(pipeline_cam2, result_queue_cam2, signal_cam2)
       

    elif sys.argv[1] == "test":
        # Load in json file
        with open("datalist.json", "r") as f:
            data = json.load(f)
        
        Thread(target=serialize_loop).start()
        predict.test(data[sys.argv[2]])




            
    # TODO: Uncomment to integrate serial
    # s.close()



'''
    Potential Optimizations:
        - Real world mapping is a bit jank rn (especially the front)
        - Interpolation (don't find the exact closest point, interpolate between the two closest points if it's not exactly at y)
        - Decrease of time between movements as it gets closer to the target
        - Dynamic FPS


'''