import cv2 as cv
import numpy as np
import queue_utils
import time 
from detect import detect_frame
TABLE_HEIGHT = 100 # TODO: in pixels
SAMPLE_TIME = 0.1 # seconds (to put into result queue)

class Pipeline2D:
    
    def __init__(self, frame_queue, result_queue, debug_queue):
        width = 1200
        height = 800
        self.myvideo=cv.VideoWriter("../out/out.avi", cv.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))

        fps = 60
        dt = 1/fps
        noise = 3
        sigmaM = 0.0001
        sigmaZ = 3*noise
        ac = dt**2/2
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.debug_queue = debug_queue

        # A : transitionMatrix
        self.A = np.array(
                [1, 0, dt, 0,
                0, 1, 0, dt,
                0, 0, 1, 0,
                0, 0, 0, 1 ]).reshape(4,4)

        # Adjust A to fit vertical velocity, maybe depth velocity?
        self.a = np.array([0, 195])

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


    def run(self):
        last_time_sampled = 0
        time_start = time.time()
        while(True):
            # if time.time() - time_start > 5:
            #     break
            key = cv.waitKey(0)
            if key == ord('q'):
                break
            # Assume there is a stream of coordinates arriving at 30 per second
            frame = self.frame_queue.get_frame()
            if frame is None:
                # print("nothing in frame queue")
                # self.debug_queue.put_frame(frame)
                continue
            # print('something in frame queue')
            # print(frame)
            # self.myvideo.write(frame)
            
            coords = detect_frame(frame)
            if coords is None:
                # self.debug_queue.put_frame(frame)
                cv.imshow("video", frame)
                continue

            xo, yo = coords
            if xo is None and yo is None:
                # self.debug_queue.put_frame(frame)
                cv.imshow("video", frame)
                continue
            

            self.mu, self.P, _ = self.kalman(self.mu, self.P, self.A, self.Q, self.B, self.a, np.array([xo, yo]), self.H, self.R)
            self.listCenterX.append(xo)
            self.listCenterY.append(yo)

            self.res += [(self.mu, self.P)]

            # Prediction
            mu2 = self.mu
            P2 = self.P
            res2 = []

            for _ in range(240):
                mu2, P2, _ = self.kalman(mu2, P2, self.A, self.Q, self.B, self.a, None, self.H, self.R)
                res2 += [(mu2, P2)]

            x_estimate = [mu[0] for mu, _ in self.res]
            y_estimate = [mu[1] for mu, _ in self.res]

            # Would be a good idea to add the uncertainty of the estimate?
            # x_uncertainty = [2 * np.sqrt(P[0, 0]) for _, P in res]
            # y_uncertainty = [2 * np.sqrt(P[1, 1]) for _, P in res]
            # z_uncertainty = [2 * np.sqrt(P[2, 2]) for _, P in res]

            x_pred = [mu2[0] for mu2, _ in res2]
            y_pred = [mu2[1] for mu2, _ in res2]
            
            # Find the x where y = TABLE_HEIGHT

            # Find the x_estimate where y_estimate is closest to TABLE_HEIGHT
            x = x_estimate[np.argmin(np.abs(np.array(y_estimate) - TABLE_HEIGHT))]
            # print(f"Predicted x: {x}, y: {TABLE_HEIGHT}")
            
            cv.circle(frame, (int(x), int(TABLE_HEIGHT)), 5, (0, 0, 255), -1)
            cv.imshow("video", frame)
            # self.debug_queue.put_frame(frame)
            # Push to result queue every SAMPLE_TIME seconds
            if time.time() - last_time_sampled > SAMPLE_TIME:
                last_time_sampled = time.time()
                self.result_queue.put_coord((x, TABLE_HEIGHT))

            # with open("../out/3d_pred", "w") as f:
            #     for i in range(len(x_estimate)):
            #         f.write(f"{x_estimate[i]} {y_estimate[i]} - {x_pred[i]} {y_pred[i]} {z_pred[i]}\n")
            # Would be a good idea to add the uncertainty of the prediction?
            # x_pred_uncertainty = [2 * np.sqrt(P2[0, 0]) for _, P2 in res2]
            # y_pred_uncertainty = [2 * np.sqrt(P2[1, 1]) for _, P2 in res2]
            # z_pred_uncertainty = [2 * np.sqrt(P2[2, 2]) for _, P2 in res2]

            # Draw the trajectory
            
        
            # Output:
            # listCenterX, listCenterY, listCenterZ (For where the ball is in the frame at each time step)
            # x_estimate, y_estimate, z_estimate (For where the ball is in next step)
            # x_pred, y_pred, z_pred (For where the ball is in the future for the next 240 iterations)



    