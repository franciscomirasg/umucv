# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Francisco Miras García <francisco.mirasg@gmail.com>
# version ='1.0'
# ---------------------------------------------------------------------------
"""
# Codigo para el ejercicio ACTIVIDAD
1.- Construye un detector de movimiento en una región de interés de la imagen marcada manualmente.
2.- Guarda 2 ó 3 segundos de la secuencia detectada en un archivo de vídeo.
Opcional: muestra el objeto seleccionado anulando el fondo.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import cv2 as cv
import numpy as np
from umucv.stream import autoStream, putText
from umucv.util import ROI
from collections import deque


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

TIME_OF_RECORDING = 2 # Tiempo de guardado en segundos
FPS = 30
TOTAL_FRAMES_SAVE = TIME_OF_RECORDING * FPS
SUAVIZAR_MEDIA = 5
UMBRAL_INICIAL = 25

cv.namedWindow('input')

# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class ControlAct:
    _region: ROI
    saved_trozo: np.ndarray
    last_frames: deque
    last_mean: deque
    umbral: float

    def __init__(self, roi=None):
        if roi is None:
            raise Exception('ROI no valido')
        self.region = roi
        self.saved_trozo = None
        self.last_frames = deque(maxlen=TOTAL_FRAMES_SAVE)
        self.last_mean = deque(maxlen=SUAVIZAR_MEDIA)
        self.umbral = UMBRAL_INICIAL/1000
        print(f'Umbral ajustado a {self.umbral}')

    def reset_trozo(self):
        self.saved_trozo = None
        self.last_frames.clear()
        self.last_mean.clear()

data = ControlAct(ROI('input'))
# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def update_umbral(v):
    v = max(1, v)
    data.umbral = v/1000
    print(f'Umbral ajustado a {data.umbral}')


def bgr2gray(x):
    return cv.cvtColor(x, cv.COLOR_BGR2GRAY).astype(float) / 255


# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

cv.createTrackbar('Umbral', 'input', 25, 1000 ,update_umbral)

# ---------------------------------------------------------------------------
# CODE
# ---------------------------------------------------------------------------

for key, frame in autoStream():
    if data.region.roi:
        [x1, y1, x2, y2] = data.region.roi
        recorte = frame[y1:y2, x1:x2]

        if key == ord('c'):
            if data.saved_trozo is not None:
                data.reset_trozo()
            data.saved_trozo = recorte
            cv.imshow('Trozo Seleccionado', data.saved_trozo)

        if data.saved_trozo is not None:
            diff = cv.absdiff(data.saved_trozo, recorte)
            diff = bgr2gray(diff)
            mean = np.mean(diff)
            means = mean
            putText(diff, f'Mean = {np.round(mean, 4)}', orig=(5, 16))
            putText(diff, f'Mean(t) = {np.round(means, 4)}', orig=(5, 16*2))

            if len(data.last_mean) > 0:
                means = np.mean(data.last_mean)

            if np.abs(means-mean) <= data.umbral:
                data.last_mean.append(mean)

            else:
                putText(diff, 'ALERT', orig=(5, y2-16))
                print('Actividad detectada')

            cv.imshow('Diferencia', diff)



        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2 - x1 + 1}x{y2 - y1 + 1}', orig=(x1, y1 - 8))
    # Mostrar Principal
    cv.imshow('input', frame)

cv.destroyAllWindows()
