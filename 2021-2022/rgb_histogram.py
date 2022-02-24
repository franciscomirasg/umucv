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

for key, frame in autoStream():
    if region.roi:
        [x1, y1, x2, y2] = region.roi
        recorte = frame[y1:y2, x1:x2]
        # B = 0, G = 1, R = 2
        b, g, r = cv.split(recorte)
        hb, bb = np.histogram(b, np.arange(0, 257, 4))
        hg, bg = np.histogram(g, np.arange(0, 257, 4))
        hr, br = np.histogram(r, np.arange(0, 257, 4))

        xsb = 2 * bb[1:]
        i = y2 - y1
        ysb = i - hb * (i / hb.max())
        xysb = np.array([xsb, ysb]).T.astype(int)

        xsg = 2 * bg[1:]
        ysg = i - hg * (i / hg.max())
        xysg = np.array([xsg, ysg]).T.astype(int)

        xsr = 2 * br[1:]
        ysr = i - hr * (i / hr.max())
        xysr = np.array([xsr, ysr]).T.astype(int)

        cv.polylines(recorte, [xysb], isClosed=False, color=(255, 0, 0), thickness=2)
        cv.polylines(recorte, [xysg], isClosed=False, color=(0, 255, 0), thickness=2)
        cv.polylines(recorte, [xysr], isClosed=False, color=(0, 0, 255), thickness=2)

        if key == ord('c'):
            pass

        if key == ord('x'):
            pass

        if trozo is not None:
            pass

        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2 - x1 + 1}x{y2 - y1 + 1}', orig=(x1, y1 - 8))

    cv.imshow('input', frame)

cv.destroyAllWindows()
