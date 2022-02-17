#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=help

import cv2 as cv
import numpy as np
from umucv.stream import autoStream, putText
from umucv.util import ROI

cv.namedWindow('input')
region = ROI('input')
trozo = None

def bgr2gray(x):
    return cv.cvtColor(x,cv.COLOR_BGR2GRAY).astype(float)/255


for key, frame in autoStream():
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        recorte = frame[y1:y2, x1:x2]
        trozo_aux = bgr2gray(recorte)

        if key == ord('c'):
            trozo = frame[y1:y2, x1:x2]
            cv.imshow('trozo', trozo)

        if key == ord('x'):
            region.roi = []

        if trozo is not None:
            diff = cv.absdiff(trozo,recorte)
            gray = bgr2gray(diff)
            mean = np.mean(gray)
            max_val = np.max(gray)

            cv.imshow('difference', diff)

            putText(frame, f'Max = {max_val}, Mean = {mean}', orig=(x1, y2 + 14))

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}',orig=(x1,y1-8))




    cv.imshow('input', frame)


cv.destroyAllWindows()
