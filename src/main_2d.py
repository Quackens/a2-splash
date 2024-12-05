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
# import serial
PAUSE_1S = "G4 P1"
GOTO_ZERO = "G1 X0 Y0"


#-----------------------------arduino setup functions--------------------------
# send XY coordinate to go to
def gcode_goto(s, x: float, y: float):
    # bounds check
    if not (-10 <= x <= 10): return -1
    elif not (-10 <= y <= 10): return -1

    gcode = 'G1 X' + str(x) + ' Y' + str(y) + '\n'
    s.write(bytes(gcode + '\n', 'utf-8'))
    grbl_out = s.readline() # Wait for grbl response with carriage return
    print(' : ' + str(grbl_out.strip()))

# send raw gcode command
def gcode_send(s, command: str):
    s.write(bytes(command + '\n', 'utf-8'))
    grbl_out = s.readline() # Wait for grbl response with carriage return
    print(' : ' + str(grbl_out.strip()))

# initialize grbl connection after opening serial comms
def grbl_init(s):
    # Wake up grbl
    print("Waking up grbl")
    s.write(bytes("\r\n\r\n", 'utf-8'))
    time.sleep(2)   # Wait for grbl to initialize 
    s.flushInput()  # Flush startup text in serial input


    print("Zeroing grbl, setting feed rate")
    gcode_send(s, '$X') # exit lockout state. initially in lockout when limit switches enabled
    gcode_send(s, 'G21') # specify millimeters (kinda... coordinates supplied in cm)
    gcode_send(s, 'G90') # absolute coordinates
    gcode_send(s, 'G17') # XY Plane
    gcode_send(s, 'G94') # units per minute feed rate mode
    gcode_send(s, '$H')  # do homing
    
    gcode_send(s, "F2000") # Set feed rate. for some reason, has to be low right after homing
    gcode_send(s, GOTO_ZERO) # Go to zero zero
    gcode_send(s, PAUSE_1S)
    gcode_send(s, "F3750") # Set feed rate for normal operation
    return


def serialize_loop():
    # global s 
    global result_queue
    # Taken from https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756
    # arduino = serial.Serial(port='COM7', baudrate= 115200, timeout=.1)
    from pipeline_2d import CUP_LEFT_X, CUP_RIGHT_X, CUP_CENTRE_X
    print("waiting1 ")
    while True:
        print("waiting2")
        coord = result_queue.get_coord()
        if coord is None:
            continue
        x, y = coord
        # Normalize the x, y coordinates
        if x > CUP_RIGHT_X or x < CUP_LEFT_X:
            print(f"Invalid x coordinate: {x}")
            continue

        if x > CUP_CENTRE_X:
            x = (x) / (CUP_RIGHT_X - CUP_CENTRE_X) * 10
        else:
            x = (x) / (CUP_CENTRE_X - CUP_LEFT_X) * 10

        # 2D now so just hardcode
        y = 0
        # gcode_goto(s, x, y)
        # arduino.write(f"{x} {y}\n".encode())

        print(f"Sent to arduino: {x} {0}")


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
    '''
    # Open grbl serial port
    s = serial.Serial('COM7',115200)

    # initialize grbl connection
    grbl_init(s)
    '''

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
        predict.run()
    elif sys.argv[1] == "test":
        # Load in json file
        with open("datalist.json", "r") as f:
            data = json.load(f)
        
        Thread(target=serialize_loop).start()
        predict.test(data[sys.argv[2]])




            
    # TODO: Uncomment to integrate serial
    # s.close()
