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
# import serial

from pipeline_2d import Pipeline2D
from queue_utils import CameraQueue2D, CoordQueue2D, FrameQueue
from threading import Thread

# Taken from https://projecthub.arduino.cc/ansh2919/serial-communication-between-python-and-arduino-663756
# arduino = serial.Serial(port='COM7', baudrate= 115200, timeout=.1)

PAUSE_1S = "G4 P1"
GOTO_ZERO = "G1 X0 Y0"

'''
#--------------------------camera setup------------------------------
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

#----------------------------ball detection + kalman function ----------
'''

'''
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

'''
# def serialize_loop(s: serial.Serial, result_queue):
def serialize_loop():
    while True:
        coord = result_queue.get_coord()
        if coord is None:
            continue
        x, y = coord
        # gcode_goto(s, x, y)
        # arduino.write(f"{x} {y}\n".encode())
        print(f"Sent to arduino: {x} {y}")


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
    # camRgb.setVideoSize(1920, 1080)
    camRgb.setVideoSize(1200, 800)
    # camRgb.setVideoSize(640, 400)
    camRgb.setFps(60)
    xoutVideo.input.setBlocking(False)
    xoutVideo.input.setQueueSize(1)
    # Linking
    camRgb.video.link(xoutVideo.input)

    with dai.Device(pipeline) as device:
        video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
        startTime = time.monotonic()
        counter = 0
        fps = 0
        frame = None
        while True:
            # if cv2.waitKey(1) == ord('q'):
            #     break
            color = (0, 0, 255)
            videoIn = video.get()
            frame = videoIn.getCvFrame()
            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time
            cv2.putText(frame, "FPS: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color)
            # cv2.imshow("video", frame)
            # print("feeding frame")
            print("got here")
            frame_queue.put_frame(frame)

            # debug_frame = debug_queue.get_frame()
            # cv2.imshow("debug", debug_frame)


if __name__ == '__main__':
    '''
    # Open grbl serial port
    s = serial.Serial('COM7',115200)

    # initialize grbl connection
    grbl_init(s)
    '''

    # print("Now accepting user input. Enter q to exit")
    # print("Shortcut to send coordinates: c X Y")
    # print("Otherwise, send raw GCODE commands")
    # while True: # user input loop
    #     cmd = input("\nEnter command to send to grbl: ").strip()
    #     if cmd == "q": break
    #     elif cmd[0].lower() == "c": # goto X Y
    #         c, x, y = cmd.split(" ")
    #         gcode_goto(s, float(x), float(y))
    #     else: # send RAW gcode command
    #         gcode_send(s, cmd)
    #     gcode_send(s, PAUSE_1S)
    #     gcode_send(s, GOTO_ZERO)
    
    ################# Prediction Pipeline Setup #################
    frame_queue = FrameQueue()
    result_queue = CoordQueue2D()
    debug_queue = FrameQueue()
    predict = Pipeline2D(frame_queue, result_queue, debug_queue)
    


    ################# Connect to device and start pipeline #################
    # Output video
    width = 1200
    height = 800
    myvideo=cv2.VideoWriter("/out/vidout.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))



    # Main Thread: Start pipeline and get frames, and send to serializer
    # Thread(target=predict.run).start()
    Thread(target=serialize_loop).start()
    Thread(target=feed_frames).start()
    # Thread(target=predict.run).start()
    predict.run()
    # feed_frames()




            
               
    # s.close()
