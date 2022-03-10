import cv2 as cv
from umucv.stream import autoStream

for key, frame in autoStream():
    gray = cv.cvt