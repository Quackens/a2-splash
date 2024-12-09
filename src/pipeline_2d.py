import cv2 as cv
import numpy as np
import queue_utils
import time 
from detect import detect_frame
import gcode.serial_comms_gcode as serial_comms_gcode



# TODO: realign
TABLE_HEIGHT = 852 # Height of the table in pixels (WxH)

CUP_LEFT_X = 102 # Left edge of the cup in pixels (WxH)
CUP_CENTRE_X = 200 # Centre of the cup in pixels (WxH)
CUP_RIGHT_X = 270 # Right edge of the cup in pixels (WxH)

SAMPLE_TIME = 0.1 # seconds (to put into result queue)

GRAVITY = 8600

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
        self.listCenterX=[]
        self.listCenterY=[]

        self.last_prediction = None

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
        serial_comms_gcode.gcode_goto(serial, 0, 0)
        self.coord_queue.reset_queue()
        self.last_prediction = None

    
    def test(self, data_list):
        last_prediction = None
        last_time_sampled = 0
        wait_time = 1
        start_time = time.time()
        # TABLE_HEIGHT = data_list[-1][1]
        print(f"Table Height: {TABLE_HEIGHT}")
        print(f"Last coord: {data_list[-1]}")
        for coords in data_list:
            key = cv.waitKey(wait_time)
            if key == ord('q'):
                break
            elif key == ord('p'):
                # Pause
                wait_time = 0
            elif key == ord('s'):
                wait_time = 100
            canvas = np.zeros((1080,1920,3), dtype=np.uint8)
            cv.line(canvas, (CUP_LEFT_X, TABLE_HEIGHT), (CUP_RIGHT_X, TABLE_HEIGHT), (0, 255, 0), 2)
            cv.circle(canvas, (CUP_CENTRE_X, TABLE_HEIGHT), 5, (255, 255, 0), -1)
            # print(coords)
            if coords is None:
                cv.imshow("video", canvas)
                continue
            
            xo, yo = coords
            if xo is None and yo is None:
                cv.imshow("video", canvas)
                continue
            cv.circle(canvas,(xo, yo), 5, (255, 255, 0), 3)

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
            
            # Find the x where y = TABLE_HEIGHT
            for n in range(len(x_pred)): # x y predicted
                cv.circle(canvas,(int(x_pred[n]),int(y_pred[n])), 1,( 0, 0, 255))


            
            
            if time.time() - last_time_sampled > SAMPLE_TIME and time.time() - start_time > 0.4:
                # Find the x_estimate where y_estimate is closest to TABLE_HEIGHT
                x = x_pred[np.argmin(np.abs(np.array(y_pred) - TABLE_HEIGHT))]
                # print(f"Predicted x: {x}, y: {TABLE_HEIGHT}")
                last_time_sampled = time.time()
                self.result_queue.put_coord((x, TABLE_HEIGHT))
                
                if x < CUP_RIGHT_X and x > CUP_LEFT_X:
                    last_prediction = x

                cv.circle(canvas, (int(x), int(TABLE_HEIGHT)), 5, (0, 0, 255), -1)
            if last_prediction:
                cv.circle(canvas, (int(last_prediction), int(TABLE_HEIGHT)), 5, (0, 0, 255), -1)
            cv.imshow("video", canvas)

    def run(self, serial):
        
        last_time_sampled = 0
        wait_time = 1
        start_time = time.time()
        last_run = time.time()

        print(f"Table Height: {TABLE_HEIGHT}")

        while(True):
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('r'):
                self.reset(serial)
                
            if time.time() - last_run > 2:
                self.reset(serial)
            
            # Assume there is a stream of coordinates arriving at 30 per second
            coords = self.coord_queue.get_coord()


            canvas = np.zeros((1080,1920,3), dtype=np.uint8)
            cv.line(canvas, (CUP_LEFT_X, TABLE_HEIGHT), (CUP_RIGHT_X, TABLE_HEIGHT), (0, 255, 0), 2)
            cv.circle(canvas, (CUP_CENTRE_X, TABLE_HEIGHT), 5, (255, 255, 0), -1)
            # print(coords)
            if coords is None:
                cv.imshow("video", canvas)
                continue
            last_run = time.time()
            xo, yo = coords
            if xo is None and yo is None:
                cv.imshow("video", canvas)
                continue
            cv.circle(canvas,(xo, yo), 5, (255, 255, 0), 3)

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
            
            # Find the x where y = TABLE_HEIGHT
            for n in range(len(x_pred)): # x y predicted
                cv.circle(canvas,(int(x_pred[n]),int(y_pred[n])), 1,( 0, 0, 255))


            
            
            if time.time() - last_time_sampled > SAMPLE_TIME and time.time() - start_time > 0.2:
                # Find the x_estimate where y_estimate is closest to TABLE_HEIGHT
                x = x_pred[np.argmin(np.abs(np.array(y_pred) - TABLE_HEIGHT))]
                print(f"Predicted x: {x}, y: {TABLE_HEIGHT}")
                last_time_sampled = time.time()
                self.result_queue.put_coord((x, TABLE_HEIGHT))
                
                if x < CUP_RIGHT_X and x > CUP_LEFT_X:
                    self.last_prediction = x

                cv.circle(canvas, (int(x), int(TABLE_HEIGHT)), 5, (0, 0, 255), -1)
            if self.last_prediction:
                cv.circle(canvas, (int(self.last_prediction), int(TABLE_HEIGHT)), 5, (0, 0, 255), -1)
            cv.imshow("video", canvas)




class Pipeline2D_CAM2:
    
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



    