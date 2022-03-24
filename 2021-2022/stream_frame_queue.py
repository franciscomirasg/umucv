#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=help

import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np

frames = deque(maxlen=10)


for key, frame in autoStream():
    aux = cv.resize(frame, (160,140))
    frames.append(aux)

    screen = np.hstack(frames)

    cv.imshow('lastFrames', screen)


cv.destroyAllWindows()
