#!/usr/bin/env python3
from pathlib import Path
import cv2
import depthai as dai
import time

from pipeline_2d import Pipeline2D
from queue_utils import CoordQueue2D, FrameQueue
from threading import Thread
from detect import detect_frame

import json
import sys

# TODO: Uncomment to integrate serial
import serial
import random
import math

import gcode.serial_comms_gcode as serial_comms_gcode


def serialize_loop():
    global s 
    global result_queue
    from pipeline_2d import CUP_LEFT_X, CUP_RIGHT_X, CUP_CENTRE_X

    while True:

        coord = result_queue.get_coord()
        if coord is None:
            continue
        x, y = coord
        # Normalize the x, y coordinates
        if x > CUP_RIGHT_X:
            gantry_x = 10
        elif x < CUP_LEFT_X:
            gantry_x = -10
        else:
            gantry_x = (x - CUP_CENTRE_X) / 7.5


        # 2D now so just hardcode
        serial_comms_gcode.gcode_goto(s, gantry_x, 0)

        print(f"Sent to arduino: {gantry_x} {0}")



def feed_frames():
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
    Path("coords.txt").unlink(missing_ok=True)
    
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
            coord_queue.put_coord(coord)

            # Writing coordinates to file
            with open("coords.txt", "a") as f:
                if coord:
                    f.write(f"[{coord[0]}, {coord[1]}],\n")
                else:
                    f.write("null,\n")



if __name__ == '__main__':
    # TODO: Uncomment to integrate serial
    # Open grbl serial port
    s = serial.Serial('/dev/tty.usbmodem101',115200)

    # initialize grbl connection
    serial_comms_gcode.grbl_init(s)


    ################# Prediction Pipeline Setup #################
    # frame_queue = FrameQueue()
    coord_queue = CoordQueue2D()
    result_queue = CoordQueue2D()
    debug_queue = FrameQueue()
    predict = Pipeline2D(coord_queue, result_queue, debug_queue)
    


    ################# Connect to device and start pipeline #################
    # Output video
    width = 1920
    height = 1080
    myvideo=cv2.VideoWriter("/out/vidout.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))

    # Main Thread: Start pipeline and get frames, and send to serializer

    
    if sys.argv[1] == "live":
        Thread(target=serialize_loop).start()
        Thread(target=feed_frames).start()
        predict.run(s)
    elif sys.argv[1] == "test":
        # Load in json file
        with open("datalist.json", "r") as f:
            data = json.load(f)
        
        Thread(target=serialize_loop).start()
        predict.test(data[sys.argv[2]])




            
    # TODO: Uncomment to integrate serial
    s.close()
