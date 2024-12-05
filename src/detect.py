import cv2
import imutils
import numpy as np
from pathlib import Path

def detect_frame(frame):
    # print(frame)
    # orangeLower = (6, 150, 200)
    # orangeUpper = (25, 255, 255)
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

    # Trying with moments
    # cx, cy = None, None
    # moments = cv2.moments(mask)
    # if moments["m00"] != 0:
    # # Centroid (x, y)
    #     cx = int(moments["m10"] / moments["m00"])  # x-coordinate
    #     cy = int(moments["m01"] / moments["m00"])  # y-coordinate
    #     # cv2.circle(mask, (cx, cy), 20, (255, 0, 255), 1)

    # # cv2.imshow("orange", mask)
    # return cx, cy


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
    cap = cv2.VideoCapture(video_path)
    wait_time = 10
    frame = None
    Path("coords").unlink(missing_ok=True)
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
        np.save("frame", frame)
        # print(frame)
        if not ret:
            break
        centre = detect_frame(frame)
        with open("coords", "a") as f:
            if centre:
                f.write(f"[{centre[0]}, {centre[1]}],\n")
            else:
                f.write("null,\n")
        
        if not centre:
            cv2.imshow("video", frame)
            continue
        cv2.circle(frame, centre, 5, (0, 0, 255), -1)

        cv2.imshow("video", frame)
        cv2.setMouseCallback('video', click_event)

        