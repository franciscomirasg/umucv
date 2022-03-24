#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=help

import cv2          as cv
from umucv.stream import autoStream

for key,frame in autoStream():
    h,w,_ = frame.shape
    dh = h//4
    dw = w//4
    
    x1 = dw
    x2 = 3*dw
    y1 = dh
    y2 = 3*dh
    cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,0,255),thickness=2)
    
    frame2 = frame[y1:y2,x1:x2]
    # frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY) # Forma 1
    frame2 = frame2[:,:, 1] # Metodo de clase
    
    cv.imshow('input',frame)
    cv.imshow('recorte', frame2)

cv.destroyAllWindows()

