# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:35:53 2023

@author: nicko
"""

import os
import numpy as np
import cv2 as cv
import glob
'''
*PLEASE CHECK single quotes for directions throughout the initial
part of the doc*

Parameters 
'''
# these are to suit the checkerboards included in 'oneplus(NUMBER)' and 
# SA73(NUMBER) files with 3_10, do not change for now

CHESSBOARD_WIDTH = 15  # Chessboard dimensions
CHESSBOARD_HEIGHT = 10   #
RESIZE_IMAGES = True
SCALING = .4
debugmode = 0
chess_size = 16  # previously 26.47


'''
Add your path to 'oneplus(NUMBER)' to the IMAGES variable, 
use the * to include all image files within the parent folder.
(ie IMG*)  This will use create a calibration location
'''

# This is the only thing to change on this document as of 13/10 
IMAGES = glob.glob('C:/Users/skippy/.spyder-py3/Calibration_2023-09-1914_34_50/frame*')
print(IMAGES)


def resize_image(img):
    # copied from: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    width = int(img.shape[1] * SCALING)
    height = int(img.shape[0] * SCALING)
    dim = (width, height)

    # resize image
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def calibrate_camera():
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHESSBOARD_WIDTH * CHESSBOARD_HEIGHT, 3), np.float32)
    objp[:, :2] = (np.mgrid[0:CHESSBOARD_WIDTH, 0:CHESSBOARD_HEIGHT] * chess_size).T.reshape(-1, 2)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fn in IMAGES:
        print(fn)
        img = cv.imread(fn)

        if RESIZE_IMAGES:
            img = resize_image(img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        if debugmode: 
            cv.imshow('gray', gray)
            cv.waitKey()
        ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT),flags=None)
        # If found, add object points, image points (after refining them)
        print(ret)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (6, 6), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            if debugmode: 
                cv.drawChessboardCorners(img, (CHESSBOARD_WIDTH, CHESSBOARD_HEIGHT), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey()

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Estimate error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print('(K)mtx:',mtx)
    print(ret)
    print("total error: {}".format(mean_error / len(objpoints)))
    print("------")

    print("CMTX = np.array({}, dtype='float32').reshape(3, 3)".format(list(mtx.flatten())))
    print("DIST = np.array({}, dtype='float32')".format(list(dist[0])))


    img = cv.imread(IMAGES[10])
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('calibresult.png', dst)

    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow ('calibresult.png', dst)

    mean_error = 0
    for i in range(len(objpoints)):
     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
     mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )


if __name__ == '__main__':
    calibrate_camera()

    
