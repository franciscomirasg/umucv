# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Francisco Miras García <francisco.mirasg@gmail.com>
# version ='1.0'
# ---------------------------------------------------------------------------
"""
# Codigo para el ejercicio Color
1.- Construye un clasificador de objetos en base a la similitud de los histogramas de color del ROI
(de los 3 canales por separado).
Opcional: Segmentación densa por reproyección de histograma.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import cv2 as cv
import numpy as np
from umucv.stream import autoStream, putText
from umucv.util import ROI

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

MODEL_W = 128
MODEL_H = 128
MODEL_DIM = (MODEL_W, MODEL_H)
MODELS_PER_ROW = 8
cv.namedWindow('input')


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class ColorParams:
    region: ROI
    saved_trozo: np.ndarray
    patrones: list

    def __init__(self, roi=None):
        if roi is None:
            raise Exception('ROI no valido')
        self.region = roi
        self.saved_trozo = None
        self.patrones = list()

    def reset_trozo(self):
        self.saved_trozo = None


class Pattern:
    frame: np.ndarray
    color_info: tuple

    def __init__(self, frame, color_info):
        self.frame = frame
        self.color_info = color_info


data = ColorParams(ROI('input'))


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def bgr2gray(x):
    return cv.cvtColor(x, cv.COLOR_BGR2GRAY).astype(float) / 255


def gray2bgr(x):
    return cv.cvtColor(x, cv.COLOR_GRAY2BGR).astype(float) / 255


def make_histogram(c, size):
    h, b = np.histogram(c, np.arange(0, 257, 4))
    x = 2 * b[1:]
    y = size - h * (size / h.sum())
    xy = np.array([x, y]).T.astype(int)
    return xy


def make_rgb_histogram(f, size):
    blue, green, red = cv.split(f)
    blue = make_histogram(blue, size)
    green = make_histogram(green, size)
    red = make_histogram(red, size)
    return blue, green, red


def resize(f, dim):
    return cv.resize(f, dim, interpolation=cv.INTER_LINEAR)


def clear_info(patterns):
    aux = list()
    for p in patterns:
        aux.append(p.frame)
    return aux


def stack_patterns(data, n=MODELS_PER_ROW):
    aux = clear_info(data.patrones)
    result = list()
    i = 0
    aux_row = None
    for f in aux:
        if i <= 0:
            if aux_row:
                result.append(np.hstack(aux_row))
            aux_row = list()
            aux_row.append(f)
        else:
            aux_row.append(f)
        i = (i + 1) % MODELS_PER_ROW

    if aux_row:
        while len(aux_row) < MODELS_PER_ROW and len(result) > 0:
            aux_row.append(np.zeros_like(f))

        result.append(np.hstack(aux_row))

    return np.vstack(result)


def hg_intersection(hg1, hg2):
    minimo = np.minimum(hg1, hg2)
    intersect = np.true_divide(np.sum(minimo), np.sum(hg2))
    return intersect


# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CODE
# ---------------------------------------------------------------------------

for key, frame in autoStream():
    if data.region.roi:
        [x1, y1, x2, y2] = data.region.roi
        recorte = frame[y1:y2, x1:x2]
        b, g, r = make_rgb_histogram(recorte, y2 - y1)

        if key == ord('c'):
            info = (b, g, r)
            data.patrones.append(Pattern(resize(recorte, MODEL_DIM), info))

        if key == ord('x'):
            data.reset_trozo()
            data.region.roi = None
            continue

        if key == ord('r'):
            data.patrones.clear()
            try:
                cv.destroyWindow('modelos')
            except Exception:
                pass

        if len(data.patrones) > 0:
            print(data.patrones[0].color_info[0] - b)
            pass

        cv.polylines(recorte, [b], isClosed=False, color=(255, 0, 0), thickness=2)
        cv.polylines(recorte, [g], isClosed=False, color=(0, 255, 0), thickness=2)
        cv.polylines(recorte, [r], isClosed=False, color=(0, 0, 255), thickness=2)

        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2 - x1 + 1}x{y2 - y1 + 1}', orig=(x1, y1 - 8))

        if len(data.patrones) > 0:
            cv.imshow('modelos', stack_patterns(data))

    cv.imshow('input', frame)

cv.destroyAllWindows()
