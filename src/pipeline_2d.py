import cv2 as cv
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import time 
import numpy.linalg as la
import imutils

def detect(frame):
    # print(frame)
    orangeLower = (6, 150, 200)
    orangeUpper = (25, 255, 255)
    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv.inRange(hsv, orangeLower, orangeUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    # cv.imshow("orange", mask)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10 and center != None:
            return center

def kalman(x_esti,P,A,Q,B,u,z,H,R):

    x_pred = A @ x_esti + B @ u;         # B : controlMatrix -->  B @ u : gravity
    #  x_pred = A @ x_esti or  A @ x_esti - B @ u : upto
    P_pred  = A @ P @ A.T + Q;

    zp = H @ x_pred

    # si no hay observación solo hacemos predicción 
    if z is None:
        return x_pred, P_pred, zp

    epsilon = z - zp

    k = P_pred @ H.T @ la.inv(H @ P_pred @ H.T +R)

    x_esti = x_pred + k @ epsilon;
    P  = (np.eye(len(P))-k @ H) @ P_pred;
    return x_esti, P, zp


# Input video
cap = cv.VideoCapture("../videos/newcrop_flipped.mp4")


# Output video
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
myvideo=cv.VideoWriter("../out/vidout.avi", cv.VideoWriter_fourcc('M','J','P','G'), 30, (int(width),int(height)))




###################### Kalman Initialization ########################

fps = 60
dt = 1/fps
# t = np.arange(0,2.01,dt)
noise = 3

# A : transitionMatrix
A = np.array(
    [1, 0, dt, 0,
    0, 1, 0, dt,
    0, 0, 1, 0,
    0, 0, 0, 1 ]).reshape(4,4)

# Adjust A to fit vertical velocity
a = np.array([0, 195])
# B : controlMatrix
B = np.array(
    [dt**2/2, 0,
    0, dt**2/2,
    dt, 0,
    0, dt ]).reshape(4,2)
# H : measurementMatrix
H = np.array(
    [1,0,0,0,
    0,1,0,0]).reshape(2,4)

# x, y, vx, vy
mu = np.array([0,0,0,0])
P = np.diag([1000,1000,1000,1000])**2
res=[]

sigmaM = 0.0001
sigmaZ = 3*noise

Q = sigmaM**2 * np.eye(4)   # processNoiseCov
R = sigmaZ**2 * np.eye(2)   # measurementNoiseCov
listCenterX=[]
listCenterY=[]
listpuntos=[]

add_count = 0
while(True):
    
    ret, frame = cap.read()
    if not ret: break
    xo, yo = None, None
    if (frame is None): 
        # cv.imshow('Frame', frame)
        myvideo.write(frame)
        continue
    # print(frame)
    coords = detect(frame)
    if coords == None:
        # cv.imshow('Frame', frame)
        myvideo.write(frame)
        continue
    else:
        xo, yo = coords



    if xo is not None and yo is not None:
        mu,P,pred= kalman(mu,P,A,Q,B,a,np.array([xo,yo]),H,R)
        listCenterX.append(xo)
        listCenterY.append(yo)

        res += [(mu,P)]

        ##### Prediction #####
        mu2 = mu
        P2 = P
        res2 = []

        for _ in range(120*2):
            mu2,P2,pred2= kalman(mu2,P2,A,Q,B,a,None,H,R)
            res2 += [(mu2,P2)]
            
        xe = [mu[0] for mu,_ in res]
        xu = [2*np.sqrt(P[0,0]) for _,P in res]
        ye = [mu[1] for mu,_ in res]
        yu = [2*np.sqrt(P[1,1]) for _,P in res]
        
        xp=[mu2[0] for mu2,_ in res2]
        yp=[mu2[1] for mu2,_ in res2]

        xpu = [2*np.sqrt(P[0,0]) for _,P in res2]
        ypu = [2*np.sqrt(P[1,1]) for _,P in res2]

        for n in range(len(listCenterX)): # Track centre of ball
            cv.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)


        for n in [-1]:#range(len(xe)): # Estimate ball
            incertidumbre=(xu[n]+yu[n])/2
            cv.circle(frame,(int(xe[n]),int(ye[n])),int(incertidumbre),(255, 255, 0),1)

        for n in range(len(xp)): # estimate trajectory
            incertidumbreP=(xpu[n]+ypu[n])/2
            # cv.circle(frame,(int(xp[n]),int(yp[n])),int(incertidumbreP),(0, 0, 255))
            cv.circle(frame,(int(xp[n]),int(yp[n])),1,(0, 0, 255))

        # if(len(listCenterY)>40):
        #     if ((listCenterX[-3] > listCenterX[-2]) and (listCenterX[-1] > listCenterX[-2])) or (abs(listCenterY[-1] - listCenterY[-2]) < 5) and (abs(listCenterY[-2] - listCenterY[-3]) < 5) :
        #         print("REBOTE")
        #         listCenterY=[]
        #         listCenterX=[]
        #         listpuntos=[]
        #         res=[]
        #         mu = np.array([0,0,0,0])
        #         P = np.diag([100,100,100,100])**2

    # time.sleep(0.1)
    # cv.imshow('ColorMask',colorMask)
    # #cv.imshow(’ColorMask’,cv.resize(colorMask,(800,600)))
    # cv.imshow('mask', bgs)
    #cv.imshow(’Frame’,cv.resize(frame,(800,600)))
    # cv.imshow('Frame', frame)
    myvideo.write(frame)

cv.destroyAllWindows()