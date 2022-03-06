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
import enum

import cv2 as cv
import numpy as np
from umucv.stream import autoStream, putText
from umucv.util import ROI, Video
from collections import deque

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

TIME_OF_RECORDING = 2  # Tiempo de guardado en segundos
FPS = 30
TOTAL_FRAMES_SAVE = TIME_OF_RECORDING * FPS
SUAVIZAR_MEDIA = 10
UMBRAL_INICIAL_DETEC = 10
UMBRAL_INICIAL_RECORTE = 100

cv.namedWindow('input')


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class Estado(enum.Enum):
    ACTIVITY = 1
    END = 2


class ControlAct:
    region: ROI
    saved_trozo: np.ndarray
    last_frames: deque
    last_mean: deque
    umbral_deteccion: float
    umbral_recorte: float
    video = None
    estado: Estado

    def __init__(self, roi=None):
        if roi is None:
            raise Exception('ROI no valido')
        self.region = roi
        self.saved_trozo = None
        self.last_frames = deque(maxlen=TOTAL_FRAMES_SAVE)
        self.last_mean = deque(maxlen=SUAVIZAR_MEDIA)
        self.umbral_deteccion = UMBRAL_INICIAL_DETEC / 1000
        self.estado = Estado.END

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
    data.umbral_deteccion = v / 1000
    print(f'Umbral de deteccion ajustado a {data.umbral_deteccion}')


def update_umbral_recorte(v):
    v = max(1, v)
    data.umbral_recorte = v / 1000
    print(f'Umbral de recorte ajustado a {data.umbral_deteccion}')


def bgr2gray(x):
    return cv.cvtColor(x, cv.COLOR_BGR2GRAY).astype(float) / 255


def gray2bgr(x):
    r = cv.cvtColor(np.float32(x), cv.COLOR_GRAY2BGR).astype(float)
    return r


def start_video(data: ControlAct):
    if data.video or len(data.last_frames) < 1:
        return
    data.video = Video(fps=FPS, codec="MJPG", ext="avi")
    data.video.ON = True
    for f in data.last_frames:
        data.video.write(f)
    data.last_frames.clear()


def continue_video(data: ControlAct, f):
    if data.video:
        data.video.write(f)


def stop_video(data: ControlAct):
    data.video.ON = False
    data.video.release()
    data.video = None


# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

cv.createTrackbar('Umbral Deteccion', 'input', UMBRAL_INICIAL_DETEC, 1000, update_umbral)
cv.createTrackbar('Umbral Recorte', 'input', UMBRAL_INICIAL_RECORTE, 1000, update_umbral_recorte)

# ---------------------------------------------------------------------------
# CODE
# ---------------------------------------------------------------------------

for key, frame in autoStream():
    if data.region.roi:
        [x1, y1, x2, y2] = data.region.roi
        recorte = frame[y1:y2, x1:x2]

        if key == ord('x'):
            data.region.roi = None
            data.reset_trozo()

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

            if len(data.last_mean) >= SUAVIZAR_MEDIA:
                means = np.mean(data.last_mean)

            if np.abs(means - mean) <= data.umbral_deteccion:
                data.last_mean.append(mean)
                data.last_frames.append(recorte)
                if data.estado is not Estado.END:
                    stop_video(data)
                    print('Fin actividad')
                    data.estado = Estado.END
                    try:
                        cv.destroyWindow('mascara')
                        cv.destroyWindow('objeto')
                    except Exception:
                        pass
            else:
                mask = diff > data.umbral_recorte
                cv.imshow('mascara', mask.astype(float))
                mask = gray2bgr(mask)
                objeto = np.zeros_like(recorte)
                np.copyto(objeto, recorte, where=mask == 1)
                cv.imshow('objeto', objeto)
                putText(diff, 'ALERT', orig=(5, diff.shape[0] - 5))

                if data.estado is Estado.END:
                    print('Actividad detectada')
                    start_video(data)
                    data.estado = Estado.ACTIVITY
                elif data.estado is Estado.ACTIVITY:
                    continue_video(data, recorte)

            putText(diff, f'Mean(t) = {np.round(means, 4)}', orig=(5, 16 * 2))
            putText(diff, f'Mean = {np.round(mean, 4)}', orig=(5, 16))
            cv.imshow('Diferencia', diff)

            data.last_frames.append(recorte)

        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2 - x1 + 1}x{y2 - y1 + 1}', orig=(x1, y1 - 8))
    # Mostrar Principal
    cv.imshow('input', frame)

cv.destroyAllWindows()
