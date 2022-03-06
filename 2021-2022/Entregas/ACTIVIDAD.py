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
FPS = 30  # FPS de la camara
TOTAL_FRAMES_SAVE = TIME_OF_RECORDING * FPS  # Tamaño de frames para guardar el tiempo requerido
SUAVIZAR_MEDIA = 10  # Numero de medias como referecia
'''
Se tiene en cuenta el valor medio de las ultimas medias para suavizar las diferencias en situacion de iluminación
variable, natural o con una camara con flickering o autofocus
'''

UMBRAL_INICIAL_DETEC = 10  # Valor inicial del umbral de detección
UMBRAL_INICIAL_RECORTE = 100  # Valor inicial del umbral de recorte

cv.namedWindow('input')


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class Estado(enum.Enum):  # Enumerado con los estados
    ACTIVITY = 1
    END = 2


class ControlAct:  # Clase contenedora con las variables usadas durante la ejecucion
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

    def start_video(self):  # Inicia el video y guarda los ultimos x segundos de inmediato
        if self.video or len(self.last_frames) < 1:
            return
        self.video = Video(fps=FPS, codec="MJPG", ext="avi")
        self.video.ON = True
        for f in self.last_frames:
            self.video.write(f)
        self.last_frames.clear()

    def continue_video(self, f):  # Añade el frame actual
        if self.video:
            self.video.write(f)

    def stop_video(self):  # Detiene el video
        self.video.ON = False
        self.video.release()
        self.video = None


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


# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

cv.createTrackbar('Umbral Deteccion', 'input', UMBRAL_INICIAL_DETEC, 1000, update_umbral)
cv.createTrackbar('Umbral Recorte', 'input', UMBRAL_INICIAL_RECORTE, 1000, update_umbral_recorte)

# ---------------------------------------------------------------------------
# CODE
# ---------------------------------------------------------------------------

for key, frame in autoStream():
    if data.region.roi:  # Si se ha seleccionado una región, la recortamos
        [x1, y1, x2, y2] = data.region.roi
        recorte = frame[y1:y2, x1:x2]

        if key == ord('x'):  # Pulsar x para eleminar región
            data.region.roi = None
            data.reset_trozo()

        if key == ord('c'):  # Pulsar c para capturar la región
            if data.saved_trozo is not None:
                data.reset_trozo()
            data.saved_trozo = recorte
            cv.imshow('Trozo Seleccionado', data.saved_trozo)

        if data.saved_trozo is not None:  # Si hay trozo guardado, empezamos con la vigilancia
            diff = cv.absdiff(data.saved_trozo, recorte)  # Calculamos la diferencia del actual con el guardado
            diff = bgr2gray(diff)
            mean = np.mean(diff)
            means = mean  # La media nos da una aproximación de cuanto ha cambiado con la de referncia

            if len(data.last_mean) >= SUAVIZAR_MEDIA:  # Si hay suficientes medias las suavizamos conforme el tiempo
                means = np.mean(data.last_mean)

            if np.abs(means - mean) <= data.umbral_deteccion:  # Si la variación no es suficiente grande.
                data.last_mean.append(mean)
                data.last_frames.append(recorte)    # Actualizamos las medias y guardamos frame
                if data.estado is not Estado.END:   # Si venimos de un estado de actividad
                    data.stop_video()               # Detenemos grabación y destruimos ventanas residuales
                    print('Fin actividad')
                    data.estado = Estado.END
                    try:
                        cv.destroyWindow('mascara')
                        cv.destroyWindow('objeto')
                    except Exception:
                        pass
            else:   # Si Hay diferencia
                mask = diff > data.umbral_recorte # Creamos mascara
                cv.imshow('mascara', mask.astype(float))
                mask = gray2bgr(mask)
                objeto = np.zeros_like(recorte) # Recortamos la mascara
                np.copyto(objeto, recorte, where=mask == 1)
                cv.imshow('objeto', objeto)
                putText(diff, 'ALERT', orig=(5, diff.shape[0] - 5)) # Notificamos la alerta

                if data.estado is Estado.END: # Si no estamos grabando, empezamos
                    print('Actividad detectada')
                    data.start_video()
                    data.continue_video(recorte)
                    data.estado = Estado.ACTIVITY
                elif data.estado is Estado.ACTIVITY:    # Continuamos grabando
                    data.continue_video(recorte)

            putText(diff, f'Mean(t) = {np.round(means, 4)}', orig=(5, 16 * 2))
            putText(diff, f'Mean = {np.round(mean, 4)}', orig=(5, 16))
            cv.imshow('Diferencia', diff)

            data.last_frames.append(recorte)

        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2 - x1 + 1}x{y2 - y1 + 1}', orig=(x1, y1 - 8))
    # Mostrar Principal
    cv.imshow('input', frame)

cv.destroyAllWindows()
