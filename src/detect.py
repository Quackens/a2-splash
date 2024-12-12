import cv2
import imutils
import numpy as np
from pathlib import Path
import sys
import depthai as dai

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()





def detect_frame(frame):

    orangeLower = (3, 150, 150)
    orangeUpper = (30, 255, 255)


    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


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
            return center
        


def detect_frame_2(frame):
    # print(frame)
    # orangeLower = (6, 150, 200)
    # orangeUpper = (25, 255, 255)
    orangeLower = (3, 150, 150)
    orangeUpper = (30, 255, 255)


    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Combine with background subtraction
    fgmask = fgbg.apply(frame)

    mask = cv2.bitwise_and(mask, mask, mask=fgmask)

    # Trying with moments

    cx, cy = None, None
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
    # Centroid (x, y)
        cx = int(moments["m10"] / moments["m00"])  # x-coordinate
        cy = int(moments["m01"] / moments["m00"])  # y-coordinate
        cv2.circle(mask, (cx, cy), 20, (255, 0, 255), 1)

    # cv2.imshow("orange", mask)
    if cx is None:
        return None
    return cx, cy

def click_event(event, x, y, flags, params): 
    global frame
    img = frame
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        pixel = img[y, x]
        print(f"X: {x}, Y: {y}, Pixel: {pixel}")

# Testing pipeline
if __name__ == "__main__":

    video_path = "../out/combined.mp4"
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(2)
    wait_time = 10
    frame = None
    # Path("coords").unlink(missing_ok=True)

    if len(sys.argv) < 2:
        print("Usage: python detect.py -side/front")
        sys.exit(1)
    elif sys.argv[1] == "-side":
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

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            

            video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
            while True:
                key = cv2.waitKey(wait_time)
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    wait_time -= 5
                    print(f"speed: {wait_time}")
                elif key == ord('m'):
                    wait_time += 5
                    print(f"speed: {wait_time}")
                elif key == ord('p'):
                    # Pause
                    wait_time = 0

                videoIn = video.get()
                frame = videoIn.getCvFrame()

                centre = detect_frame_2(frame)
                
                if not centre:
                    cv2.imshow("video", frame)
                    continue
                cv2.circle(frame, centre, 20, (0, 0, 255), -1)

                cv2.imshow("video", frame)
                cv2.setMouseCallback('video', click_event)

    else:
        while True:
            
            key = cv2.waitKey(wait_time)
            if key == ord('q'):
                break
            elif key == ord('n'):
                wait_time -= 5
                print(f"speed: {wait_time}")
            elif key == ord('m'):
                wait_time += 5
                print(f"speed: {wait_time}")
            elif key == ord('p'):
                # Pause
                wait_time = 0
            
            
            ret, frame = cap.read()

            if not ret:
                break
            centre = detect_frame(frame)
            
            if not centre:
                cv2.imshow("video", frame)
                continue
            cv2.circle(frame, centre, 20, (0, 0, 255), -1)

            cv2.imshow("video", frame)
            cv2.setMouseCallback('video', click_event)

        