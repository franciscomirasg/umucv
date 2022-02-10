#!/usr/bin/env python
import math

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import putText
from collections import deque

# points = list()
points = deque(maxlen=2)


class Point:
    """
    Declaraciom de un punto en dos dimensiones
    x: coedenada x
    y: cordenada y
    """
    x, y = 0, 0

    def __init__(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    @property
    def cords(self) -> tuple[int, int]:
        """
        Devuelve una tupla con las coordenadas (x,y)
        """
        return (self.x, self.y)

    def __str__(self):
        return f'({self.x}, {self.y})'

    @staticmethod
    def half_point(p1, p2):
        """
        Devuelve el punto medio
        """
        aux = p1 + p2
        return Point(aux.x / 2, aux.y / 2)

    def half(self, other):

        """
        Devuelve el punto medio
        """
        return Point.half_point(self,other)


def dst_px(p1: Point, p2: Point):
    x = (p2.x - p1.x)
    x = math.pow(x, 2)
    y = (p2.y - p1.y)
    y = math.pow(y, 2)
    aux = math.sqrt(x+y)
    return round(aux, 3)


def manejador(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)
        points.append(Point(x, y))


cv.namedWindow("webcam")
cv.setMouseCallback("webcam", manejador)

for key, frame in autoStream():
    for p in points:
        cv.circle(frame, p.cords, 1, color=(0, 0, 255), thickness=-1)
        # putText(frame, str(p), p.cords)
    if len(points) > 1 and points[0] is not None and points[1] is not None:
        putText(frame, str(dst_px(points[0], points[1])) + " px", Point.half_point(points[0], points[1]).cords)
        cv.line(frame, points[0].cords, points[1].cords, color=(0, 0, 255), thickness=1)

    cv.imshow('webcam', frame)

cv.destroyAllWindows()

quit(0)
