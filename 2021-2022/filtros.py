import cv2 as cv
import numpy as np
from umucv.stream import autoStream


class data:
    _sigma: int
    _sigmacol: int
    _radx: int
    filter = None

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = max(1, value)

    @sigma.getter
    def sigma(self):
        return self._sigma / 10

    @property
    def sigmaCol(self):
        return self._sigmacol

    @sigmaCol.setter
    def sigmaCol(self, value):
        self._sigmacol = value

    @sigmaCol.getter
    def sigmaCol(self):
        return self._sigmacol / 10

    @property
    def radx(self):
        return self._radx

    @radx.setter
    def radx(self, value):
        self._radx = value * 2 + 1


values = data()


def update_sigma(v):
    values.sigma = v


def update_sigmaCol(v):
    values.sigmaCol = v


def update_radx(v):
    values.radx = v


def gaus(src, params):
    return cv.GaussianBlur(src, (0, 0), params.sigma)


def box(src, params):
    return cv.boxFilter(src, -1, (values.radx, values.radx))


cv.namedWindow("Filtros")
cv.createTrackbar("sigma", "Filtros", 1, 200, update_sigma)
cv.createTrackbar("sigmacol", "Filtros", 1, 100, update_sigmaCol)
cv.createTrackbar("radx", "Filtros", 1, 100, update_radx)

filtros = dict()
filtros['gaus'] = gaus
filtros['box'] = box

values.filter = filtros['gaus']
for key, frame in autoStream():
    if key == ord('g'):
        values.filter = filtros['gaus']

    if key == ord('b'):
        values.filter = filtros['box']

    cv.imshow('Filtros', values.filter(frame, values))

cv.destroyAllWindows()
