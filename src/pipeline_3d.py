import cv2 as cv
import numpy as np
from camera_queue import CameraQueue

class Pipeline3D:
    
    def __init__(self, queue):
        fps = 60
        dt = 1/fps
        noise = 3
        sigmaM = 0.0001
        sigmaZ = 3*noise
        ac = dt**3/3

        self.queue = queue

        # A : transitionMatrix
        self.A = np.array([
            1, 0, 0, dt, 0, 0,
            0, 1, 0, 0, dt, 0,
            0, 0, 1, 0, 0, dt,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1,
        ]).reshape(6,6)

        # Adjust A to fit vertical velocity, maybe depth velocity?
        self.a = np.array([0, 195, 0]) # TODO: Maybe tune the third parameter?

        # B : controlMatrix
        self.B = np.array([
            ac, 0, 0,
            0, ac, 0,
            0, 0, ac,
            dt, 0, 0,
            0, dt, 0,
            0, 0, dt
        ]).reshape(6,3)

        # H : measurementMatrix
        self.H = np.array([
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
        ]).reshape(3,6)

        # x, y, z, vx, vy, vz
        self.mu = np.array([0, 0, 0, 0, 0, 0])
        self.P = 1000 ** 2 * np.eye(6)
        self.res=[]

        self.Q = sigmaM**2 * np.eye(6)   # processNoiseCov
        self.R = sigmaZ**2 * np.eye(3)   # measurementNoiseCov
        self.listCenterX=[]
        self.listCenterY=[]
        self.listCenterZ=[]

    def kalman(x_esti,P,A,Q,B,u,z,H,R):
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
        while(True):
            # Assume there is a stream of coordinates arriving at 30 per second
            xo, yo, zo = self.queue.get_next()

            if xo is None and yo is None and zo is None:
                continue

            self.mu, self.P, _ = self.kalman(self.mu, self.P, self.A, self.Q, self.B, self.a, np.array([xo, yo, zo]), self.H, self.R)
            self.listCenterX.append(xo)
            self.listCenterY.append(yo)
            self.listCenterZ.append(zo)

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
            z_estimate = [mu[2] for mu, _ in self.res]

            # Would be a good idea to add the uncertainty of the estimate?
            # x_uncertainty = [2 * np.sqrt(P[0, 0]) for _, P in res]
            # y_uncertainty = [2 * np.sqrt(P[1, 1]) for _, P in res]
            # z_uncertainty = [2 * np.sqrt(P[2, 2]) for _, P in res]

            x_pred = [mu2[0] for mu2, _ in res2]
            y_pred = [mu2[1] for mu2, _ in res2]
            z_pred = [mu2[2] for mu2, _ in res2]

            # Would be a good idea to add the uncertainty of the prediction?
            # x_pred_uncertainty = [2 * np.sqrt(P2[0, 0]) for _, P2 in res2]
            # y_pred_uncertainty = [2 * np.sqrt(P2[1, 1]) for _, P2 in res2]
            # z_pred_uncertainty = [2 * np.sqrt(P2[2, 2]) for _, P2 in res2]

            # Draw the trajectory
            
            # Output:
            # listCenterX, listCenterY, listCenterZ (For where the ball is in the frame at each time step)
            # x_estimate, y_estimate, z_estimate (For where the ball is in next step)
            # x_pred, y_pred, z_pred (For where the ball is in the future for the next 240 iterations)

    