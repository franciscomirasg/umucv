#!/usr/bin/env python

# > ./stream.py
# > ./stream.py --dev=help

import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np

frames = deque(maxlen=10*5)

def chunks(L, n):
    return np.array_split(L,n)

def assamble(M):
    aux = list()
    for r in M:
        aux.append(np.hstack(r))

    return np.vstack(aux)

for key, frame in autoStream():
    aux = cv.resize(frame, (160,140))
    frames.append(aux)

    if len(frames) < frames.maxlen:
        continue

    screen = chunks(frames, 5)
    screen = assamble(screen)
    cv.imshow('lastFrames', screen)


cv.destroyAllWindows()
