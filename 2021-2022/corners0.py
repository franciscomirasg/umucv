import sys

import cv2 as cv
from umucv.stream import autoStream
import numpy as np
from umucv.util import putText
import time

'''
Funciones Auxiliares
'''

GAUSS_LEVEL = 5
debug = True
cv.namedWindow('input')


def gauss(frame, level):
    """
    Suavizado
    """
    return cv.GaussianBlur(frame, (0, 0), level)


# Derivadas horizontal y vertical
def grad(frame):
    """
    Gradiente
    """
    gx = cv.Sobel(frame, -1, 1, 0)
    gy = cv.Sobel(frame, -1, 0, 1)
    return gx, gy


# Calculo de máximos locales
def nms(frame, t=0.1, umbral=(5, 5)):
    """
    Calculo los máximos locales,
    devuelve una matriz de
    """
    m = cv.dilate(frame, np.ones(umbral, np.uint8))  # filtro de máximo, genera los maximos locales
    h = np.max(m)  # Nos quedamos con los maximos locales, más sustanciales
    return (frame == m) & (frame > t * h)


def update_gauss(x):
    global GAUSS_LEVEL
    GAUSS_LEVEL = max(1, x)


cv.createTrackbar('Gauss Level', 'input', 5, 100, update_gauss)
cv.setTrackbarMin('Gauss Level', 'input', 1)

for key, frame in autoStream():
    # Procesamos en monocromo
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)

    t0 = time.time()

    # Calculo de los gradientes
    gx, gy = grad(gauss(gray, 2))
    if debug:
        cv.imshow('debugX', gx / gx.max())
        cv.imshow('debug Y', gy / gy.max())

    # Preparamos la matrix de covarianza
    # los elementos de la matriz de covarianza Filter box hace media, gausiano mas gradual
    gx2 = gx * gx
    gy2 = gy * gy
    xyg = gx * gy

    # sus medias en un entorno sigmas, matrices covar
    sx2 = gauss(gx2, GAUSS_LEVEL)
    sy2 = gauss(gy2, GAUSS_LEVEL)
    sxy = gauss(xyg, GAUSS_LEVEL)

    if debug:
        cv.imshow('debug sX', sx2 / sx2.max())
        cv.imshow('debug sy', sy2 / sy2.max())

    # Calculamos lamda min - FIXME la raiz cuadrada requiere guardas
    try:
        lmin = sx2 + sy2 - np.sqrt(sx2 ** 2 + sy2 ** 2 + 4 * sxy ** 2 - 2 * sx2 * sy2)  # Calculo de lamda min
    except RuntimeWarning as e:
        print(e, file=sys.stderr)
        print(f'Values: sx2² = {sx2**2}')

    if debug:
        cv.imshow('lambda min', lmin / lmin.max())

    # extraemos los picos de respuesta
    cornermask = nms(lmin)
    py, px = np.where(cornermask)
    corners = np.array([px, py]).T

    t1 = time.time()

    # y los mostramos encima de la imagen original
    for x, y in corners:
        cv.circle(frame, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1, lineType=cv.LINE_AA)

    putText(frame, f'{len(corners)} corners, {(t1 - t0) * 1000:.0f}ms')
    cv.imshow('input', frame)