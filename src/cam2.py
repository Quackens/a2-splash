import cv2
import cv2 as cv
from detect import detect_frame_2 as detect_frame
import sys
import numpy as np
import time
import gcode.serial_comms_gcode as serial_comms_gcode
import serial

GRAVITY = 8600

# In pixels
HEIGHT = 595
LEFT_BOUND = 572
RIGHT_BOUND = 724
CENTRE = 655

SAMPLE_TIME = 0.1

class Pipeline2D:
    
    def __init__(self):
        width = 1920
        height = 1080
        # self.myvideo=cv.VideoWriter("../out/out.avi", cv.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))

        fps = 57
        dt = 1/fps
        noise = 3
        sigmaM = 0.0001
        sigmaZ = 3*noise
        ac = dt**2/2

        # A : transitionMatrix
        self.A = np.array(
                [1, 0, dt, 0,
                0, 1, 0, dt,
                0, 0, 1, 0,
                0, 0, 0, 1 ]).reshape(4,4)

        # Adjust A to fit vertical velocity, maybe depth velocity?
        self.a = np.array([0, GRAVITY])

        # B : controlMatrix
        self.B = np.array(
                [dt**2/2, 0,
                0, dt**2/2,
                dt, 0,
                0, dt ]).reshape(4,2)

        # H : measurementMatrix
        self.H = np.array(
                [1, 0, 0, 0,
                0, 1, 0, 0]).reshape(2,4)

        # x, y, vx, vy
        self.mu = np.array([0, 0, 0, 0])
        self.P = 1000 ** 2 * np.eye(4)
        self.res=[]

        self.Q = sigmaM**2 * np.eye(4)   # processNoiseCov
        self.R = sigmaZ**2 * np.eye(2)   # measurementNoiseCov
        self.listCenterX=[]
        self.listCenterY=[]


    def kalman(self, x_esti, P, A, Q, B, u, z, H, R):
        # B : controlMatrix -->  B @ u : gravity
        x_pred = A @ x_esti + B @ u           
        #  x_pred = A @ x_esti or  A @ x_esti - B @ u : upto

        # TODO: changed this to Q/4 in prediction mode
        if z is None:
            P_pred  = A @ P @ A.T + Q / 4
        else:
            P_pred  = A @ P @ A.T + Q

        zp = H @ x_pred

        # If no observation, then just make a prediction 
        if z is None:
            return x_pred, P_pred, zp

        epsilon = z - zp

        k = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T +R)

        x_esti = x_pred + k @ epsilon
        P  = (np.eye(len(P))-k @ H) @ P_pred
        return x_esti, P, zp


    

    def run(self, coords):
        # Assume there is a stream of coordinates arriving at 30 per second
        # coords = self.coord_queue.get_coord()

        # canvas = np.zeros((1280,720,3), dtype=np.uint8)


        if coords is None:
            # cv.imshow("video", canvas)
            return
        last_run = time.time()
        xo, yo = coords
        if xo is None and yo is None:
            # cv.imshow("video", canvas)
            return

        # cv.circle(canvas,(xo, yo), 5, (255, 255, 0), 3)

        self.mu, self.P, _ = self.kalman(self.mu, self.P, self.A, self.Q, self.B, self.a, np.array([xo, yo]), self.H, self.R)
        self.listCenterX.append(xo)
        self.listCenterY.append(yo)

        self.res += [(self.mu, self.P)]

        # Prediction
        # print(f"x,y: {xo, yo} mu: {self.mu}")
        mu2 = self.mu
        P2 = self.P
        res2 = []

        for _ in range(240):
            mu2, P2, _ = self.kalman(mu2, P2, self.A, self.Q, self.B, self.a, None, self.H, self.R)
            res2 += [(mu2, P2)]

        x_estimate = [mu[0] for mu, _ in self.res]
        y_estimate = [mu[1] for mu, _ in self.res]

        x_pred = [mu2[0] for mu2, _ in res2]
        y_pred = [mu2[1] for mu2, _ in res2]
        # print(x_pred, y_pred)
        # Find the x where y = TABLE_HEIGHT
        # for n in range(len(x_pred)): # x y predicted
        #     cv.circle(canvas,(int(x_pred[n]),int(y_pred[n])), 1,( 0, 0, 255))
        # cv.imshow("video", canvas)
        return x_pred, y_pred

def click_event(event, x, y, flags, params): 
    global frame
    img = frame
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        print(f"X: {x}, Y: {y}")
    
# Toggles whether the throw has begun (ball must initally follow upwards trajectory)
def test_throw_began(coord):
    global last_coord
    if not last_coord:
        last_coord = coord
        return False
    if coord:
        if coord[1] < last_coord[1] + 10:
            return True
    last_coord = coord
    return False

if sys.argv[1] == "live" or sys.argv[1] == "write":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920) # set the resolution
    cap.set(4, 1080)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
else:
    cap = cv2.VideoCapture(sys.argv[1])



mode = sys.argv[1]
if mode == "write":
    myvideo = cv2.VideoWriter("front2.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (500, 1080))

predict = Pipeline2D()

last_coord = None
throw_began = False

coords = []
if mode == "write" or mode == "live":
    wait_time = 1
else:
    wait_time = 100

last_time_sampled = time.time()
start_time = time.time()

s = serial.Serial('/dev/tty.usbmodem101',115200)
serial_comms_gcode.grbl_init(s)

while True:
    ret, frame = cap.read()
    cv2.setMouseCallback('frame', click_event)
    key = cv2.waitKey(wait_time)
    if key == ord('q'):
        break   
    elif key == ord('p'):
        wait_time = 0
    

    cv2.line(frame, (LEFT_BOUND, HEIGHT), (RIGHT_BOUND, HEIGHT), (0, 255, 0), 2)


    # Only take the middle 500 pixels slide of width
    if mode == "live" or mode == "write":
        frame = frame[:, 710:1210]

    if mode == "write":
        myvideo.write(frame)
    else:
        coord = detect_frame(frame)
        print(coord)

        # Only begin prediction when the throw begins
        if throw_began:
            preds = predict.run(coord)
            if coord:
                coords.append(coord)
                
            
            for coord in coords:
                cv2.circle(frame, coord, 1, (0, 0, 255), -1)

            if preds is not None:
                x_list, y_list = preds  
                for x, y in zip(x_list, y_list):
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

                # Find the x where y = TABLE_HEIGHT
                x = x_list[np.argmin(np.abs(np.array(y_list) - HEIGHT))]
                cv2.circle(frame, (int(x), HEIGHT), 5, (255, 255, 0), -1)
                
                if time.time() - last_time_sampled > SAMPLE_TIME and time.time() - start_time > 0.2:
                    last_time_sampled = time.time()
                    if x > RIGHT_BOUND:
                        serial_comms_gcode.gcode_goto(s, 0,-10)
                    elif x < LEFT_BOUND:
                        serial_comms_gcode.gcode_goto(s, 0, 10)
                    else:
                        serial_comms_gcode.gcode_goto(s, 0, HEIGHT)
                    
        else:
            throw_began = test_throw_began(coord)

        cv2.imshow("frame", frame)
    

cap.release()
cv2.destroyAllWindows()
