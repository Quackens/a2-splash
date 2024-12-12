import cv2 as cv
import numpy as np
import queue_utils
import time 
from detect import detect_frame
import gcode.serial_comms_gcode as serial_comms_gcode


TABLE_HEIGHT = 874 # Height of the table in pixels (WxH)

CUP_LEFT_X = 120 # Left edge of the cup in pixels (WxH)
CUP_CENTRE_X = 200 # Centre of the cup in pixels (WxH)
CUP_RIGHT_X = 281 # Right edge of the cup in pixels (WxH)

SAMPLE_TIME = 0.1 # seconds (to put into result queue)
#Sample @ time  0.2, 0.4, 0.5, 0.55, 0.60, 0.65, 0.67
SAMPLE_TIMES = [0.2, 0.2, 0.1, 0.05, 0.05, 0.05, 0.02] # add to 0.6 TODO
GRAVITY = 8600
# GRAVITY = 9400


class Pipeline2D:
    
    def __init__(self, coord_queue, result_queue, debug_queue):
        width = 1920
        height = 1080
        self.myvideo=cv.VideoWriter("../out/out.avi", cv.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))

        fps = 57
        dt = 1/fps
        noise = 3
        sigmaM = 0.0001
        sigmaZ = 3*noise
        ac = dt**2/2
        self.coord_queue = coord_queue
        self.result_queue = result_queue
        self.debug_queue = debug_queue

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
        # self.listCenterX=[]
        # self.listCenterY=[]

        self.last_prediction = None
        self.last_run = time.time()
        self.run_count = 0
        self.start_time = time.time()
        self.observed_x = []
        self.observed_y = []
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

    def reset(self, serial):
        self.mu = np.array([0, 0, 0, 0])
        self.P = 1000 ** 2 * np.eye(4)
        if serial:
            serial_comms_gcode.gcode_goto(serial, 0, 0)
        self.coord_queue.reset_queue()
        self.last_prediction = None
        self.last_run = time.time()
        self.run_count = 0
        self.result_queue.reset_queue()
        if self.observed_x and self.observed_y:
            x = self.observed_x[np.argmin(np.abs(np.array(self.observed_y) - TABLE_HEIGHT))]
            if x > CUP_RIGHT_X:
                gantry_x = 10
            elif x < CUP_LEFT_X:
                gantry_x = -10
            else:
                gantry_x = (x - CUP_CENTRE_X) / ((CUP_RIGHT_X - CUP_LEFT_X) / 2) *  10
                if gantry_x < -10: gantry_x = -10
                if gantry_x > 10: gantry_x = 10
            print(f"Actual Side: {gantry_x}")
        self.observed_x = []
        self.observed_y = []

    def run(self, serial):
        
        last_time_sampled = 0
        wait_time = 1
        
        self.last_run = time.time()

        print(f"Table Height: {TABLE_HEIGHT}")

        while(True):
            

            if time.time() - self.last_run > 2:
                print("RESET side cam")
                self.reset(serial)
            # Assume there is a stream of coordinates arriving at 30 per second
            coords = self.coord_queue.get_coord()
            # print(self.coord_queue.get_length())

            canvas = np.zeros((1080,1920,3), dtype=np.uint8)
            cv.line(canvas, (CUP_LEFT_X, TABLE_HEIGHT), (CUP_RIGHT_X, TABLE_HEIGHT), (0, 255, 0), 2)
            cv.circle(canvas, (CUP_CENTRE_X, TABLE_HEIGHT), 5, (255, 255, 0), -1)

            # Debugging window
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
            # if key == ord('r'):
            #     print("RESET")
            #     self.reset(serial)


            if coords is None:
                # cv.imshow("video", canvas)
                continue

            xo, yo = coords
            if xo is None and yo is None:
                # cv.imshow("video", canvas)
                continue
            cv.circle(canvas,(xo, yo), 5, (255, 255, 0), 3)
            self.observed_x.append(xo)
            self.observed_y.append(yo)

            self.mu, self.P, _ = self.kalman(self.mu, self.P, self.A, self.Q, self.B, self.a, np.array([xo, yo]), self.H, self.R)
            # self.listCenterX.append(xo)
            # self.listCenterY.append(yo)

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
            
            # Find the x where y = TABLE_HEIGHT
            for n in range(len(x_pred)): # x y predicted
                cv.circle(canvas,(int(x_pred[n]),int(y_pred[n])), 5,( 0, 0, 255))
            
            # TODO: make sure run count is changed higher if changing to 50ms samples at the 
            if self.run_count < 7 and time.time() - last_time_sampled > SAMPLE_TIMES[self.run_count] and time.time() - self.start_time > 0.2:
                # Find the x_estimate where y_estimate is closest to TABLE_HEIGHT
                # x = x_pred[np.argmin(np.abs(np.array(y_pred) - TABLE_HEIGHT))]
                
                # x0 and x1 are the x values corresponding to the two closest y values to the TABLE_HEIGHT
                i0, i1 = np.argsort(np.abs(np.array(y_pred) - TABLE_HEIGHT))[:2]
                y0, y1 = y_pred[i0], y_pred[i1]
                x0, x1 = x_pred[i0], x_pred[i1]
                # perform linear interpolation to find x at the TABLE_HEIGHT
                if y0 != y1:
                    x = x0 + (TABLE_HEIGHT - y0) * (x1 - x0) / (y1 - y0)
                else:
                    x = x0

                # print(f"Predicted x: {x}, y: {TABLE_HEIGHT}")
                last_time_sampled = time.time()
                self.result_queue.put_coord((x, TABLE_HEIGHT))
                self.run_count += 1
                if x < CUP_RIGHT_X and x > CUP_LEFT_X:
                    self.last_prediction = x
                print(f"Side cam: {x}")
                cv.circle(canvas, (int(x), int(TABLE_HEIGHT)), 5, (0, 0, 255), -1)
            if self.last_prediction:
                cv.circle(canvas, (int(self.last_prediction), int(TABLE_HEIGHT)), 5, (0, 0, 255), -1)
            
            # cv.imshow("video", canvas)
            



class Pipeline2D_CAM2:
    
    def __init__(self):
        width = 1920
        height = 1080
        # self.myvideo=cv.VideoWriter("../out/out.avi", cv.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))

        fps = 50
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


    def reset(self):
        self.mu = np.array([0, 0, 0, 0])
        self.P = 1000 ** 2 * np.eye(4)


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

        xo, yo = coords
        if xo is None and yo is None:
            # cv.imshow("video", canvas)
            return

        # cv.circle(canvas,(xo, yo), 5, (255, 255, 0), 3)

        self.mu, self.P, _ = self.kalman(self.mu, self.P, self.A, self.Q, self.B, self.a, np.array([xo, yo]), self.H, self.R)
        # self.listCenterX.append(xo)
        # self.listCenterY.append(yo)

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



    