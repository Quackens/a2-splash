import cv2
import depthai as dai
import sys


def click_event(event, x, y, flags, params): 
    global frame
    img = frame
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        print(f"X: {x}, Y: {y}")
              

# Output video
write_video = len(sys.argv) > 1 and sys.argv[2] == '-o'
if write_video:
    width = 1920
    height = 1080
    myvideo=cv2.VideoWriter("../out/color_tune_60fps.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 60, (int(width),int(height)))



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
        videoIn = video.get()
        frame = videoIn.getCvFrame()
        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        cv2.circle(frame, (200, 815), 5, (0, 0, 255), -1)
        if write_video:
            myvideo.write(frame)
        else:
            cv2.imshow("video", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        cv2.setMouseCallback('video', click_event)
