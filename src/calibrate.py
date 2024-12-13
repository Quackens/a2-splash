import cv2
import depthai as dai
import sys
import gcode.serial_comms_gcode as serial_comms_gcode
import serial

# Usage: python3 src/calibrate.py -front
#        python3 src/calibrate.py -side

if len(sys.argv) < 2:
    print("Usage: python3 src/calibrate.py -front (-o)")
    print("       python3 src/calibrate.py -side (-o)")
    sys.exit()
        
def click_event(event, x, y, flags, params): 
    global frame
    img = frame
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        print(f"X: {x}, Y: {y}")
        print(f"Normalized X: {normalize_x(x)}")

DEFAULT_XY = 15

def normalize_x(x):
    from pipeline_2d import CUP_LEFT_X, CUP_RIGHT_X, CUP_CENTRE_X
    SIDE_LEFT_BOUND = CUP_LEFT_X
    SIDE_RIGHT_BOUND = CUP_RIGHT_X
    SIDE_CENTRE = CUP_CENTRE_X

    # Normalize the x coordinate
    if x <= SIDE_LEFT_BOUND-50:
        gantry_x = -1 * DEFAULT_XY
    elif x >= SIDE_RIGHT_BOUND+50:
        gantry_x = DEFAULT_XY


    elif SIDE_RIGHT_BOUND < x < SIDE_RIGHT_BOUND+50:
        gantry_x = 10
        # gantry_x = 5

    elif SIDE_LEFT_BOUND-50 < x < SIDE_LEFT_BOUND:
        gantry_x = -10
        # gantry_x = -5

    elif SIDE_LEFT_BOUND <= x <=SIDE_RIGHT_BOUND:
        # gantry_x = (x - SIDE_CENTRE) / 7.5

        # TODO: fix
        # gantry_x = (x - SIDE_CENTRE) / ((SIDE_RIGHT_BOUND - SIDE_LEFT_BOUND) / 2) * 10
        if x < SIDE_CENTRE:
            gantry_x = ((x - SIDE_CENTRE) / (SIDE_CENTRE - SIDE_LEFT_BOUND)) * 10
        else:
            gantry_x = ((x - SIDE_CENTRE) / (SIDE_RIGHT_BOUND - SIDE_CENTRE)) * 10
        
        if gantry_x < -10: gantry_x = -10
        if gantry_x > 10: gantry_x = 10
    else:
        gantry_x = DEFAULT_XY
    return gantry_x

def normalize_y(y):
    from cam2 import LEFT_BOUND, RIGHT_BOUND, CENTRE
    FRONT_LEFT_BOUND = LEFT_BOUND
    FRONT_RIGHT_BOUND = RIGHT_BOUND
    FRONT_CENTRE = CENTRE

    # Normalize the y coordinate    
    if y <= FRONT_LEFT_BOUND-50:
        gantry_y = -1 * DEFAULT_XY
    elif y >= FRONT_RIGHT_BOUND+50:
        gantry_y = DEFAULT_XY

    if FRONT_RIGHT_BOUND < y < FRONT_RIGHT_BOUND+50:
        gantry_y = -10
        # gantry_y = -5
    elif FRONT_LEFT_BOUND-50 < y < FRONT_LEFT_BOUND:
        gantry_y = 10
        # gantry_y = 5
    elif FRONT_LEFT_BOUND <= y <= FRONT_RIGHT_BOUND:
        
        # TODO: fix
        # gantry_y = (FRONT_CENTRE - y) / ((FRONT_RIGHT_BOUND - FRONT_LEFT_BOUND) / 2) * 10
        if y < FRONT_CENTRE:
            gantry_y = ((FRONT_CENTRE - y) / (FRONT_CENTRE - FRONT_LEFT_BOUND)) * 10
        else:
            gantry_y = ((FRONT_CENTRE - y) / (FRONT_RIGHT_BOUND - FRONT_CENTRE)) * 10

        if gantry_y < -10: gantry_y = -10
        if gantry_y > 10: gantry_y = 10
    else:
        gantry_y = DEFAULT_XY
    return gantry_y

              


# Output video
write_video = len(sys.argv) > 2 and sys.argv[2] == '-o'
if write_video:
    width = 1920
    height = 1080
    myvideo=cv2.VideoWriter("../out/color_tune_60fps.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 60, (int(width),int(height)))

if sys.argv[1] == "-side":
    s = serial.Serial('/dev/tty.usbmodem21101',115200)
    # serial_comms_gcode.grbl_init(s)
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

    from pipeline_2d import TABLE_HEIGHT, CUP_CENTRE_X, CUP_LEFT_X, CUP_RIGHT_X

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

        while True:
            videoIn = video.get()
            frame = videoIn.getCvFrame()
            # Get BGR frame from NV12 encoded video frame to show with opencv
            # Visualizing the frame on slower hosts might have overhead
            cv2.circle(frame, (CUP_CENTRE_X, TABLE_HEIGHT), 5, (0, 0, 255), -1)
            cv2.circle(frame, (CUP_LEFT_X, TABLE_HEIGHT), 5, (0, 0, 255), -1)
            cv2.circle(frame, (CUP_RIGHT_X, TABLE_HEIGHT), 5, (0, 0, 255), -1)

            cv2.line(frame, (0, 895), (1920, 895), (0, 255, 0), 2)
            if write_video:
                myvideo.write(frame)
            else:
                cv2.imshow("video", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                s.close()
                break
            elif key == ord('l'):
                serial_comms_gcode.gcode_goto(s, 10, 0)
            elif key == ord('k'):
                serial_comms_gcode.gcode_goto(s, 0, 0)
            elif key == ord('j'):
                serial_comms_gcode.gcode_goto(s, -10, 0)
            
            cv2.setMouseCallback('video', click_event)
            
else:
    from cam2 import HEIGHT, CENTRE, LEFT_BOUND, RIGHT_BOUND
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        frame = frame[:, 710:1210]
        cv2.line(frame, (0, 768), (500, 768), (0, 255, 0), 2)
        cv2.circle(frame, (CENTRE, HEIGHT), 5, (0, 0, 255), -1)
        cv2.line(frame, (LEFT_BOUND, HEIGHT), (RIGHT_BOUND, HEIGHT), (0, 255, 0), 2)
        if write_video:
            myvideo.write(frame)
        else:
            cv2.imshow("video", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.setMouseCallback('video', click_event)
