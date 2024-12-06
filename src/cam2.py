import cv2
from detect import detect_frame_2 as detect_frame
import sys

if sys.argv[1] == "live" or sys.argv[1] == "write":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(sys.argv[1])

cap.set(3, 1920) # set the resolution
cap.set(4, 1080)
cap.set(cv2.CAP_PROP_FOCUS, 0)

mode = sys.argv[1]
if mode == "write":
    myvideo = cv2.VideoWriter("front2.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (500, 1080))

coords = []
while True:
    ret, frame = cap.read()
    if mode == "write" or mode == "live":
        wait_time = 1
    else:
        wait_time = 40
    if cv2.waitKey(wait_time) == ord('q'):
        break   
    
    # Only take the middle 500 pixels slide of width
    if mode == "live" or mode == "write":
        frame = frame[:, 710:1210]

    if mode == "write":
        myvideo.write(frame)

    else:
        coord = detect_frame(frame)
        
        if coord:
            coords.append(coord)
        
        for coord in coords:
            cv2.rectangle(frame, (coord[0] - 5, coord[1] - 5), (coord[0] + 5, coord[1] + 5), (0, 0, 255), -1)


        cv2.imshow("frame", frame)
    

cap.release()
cv2.destroyAllWindows()
