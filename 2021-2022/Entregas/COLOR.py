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

MODEL_W = 128  # Anchura del modelo a guardar
MODEL_H = 128  # Altura del modelo a guardar
MODEL_DIM = (MODEL_W, MODEL_H)
MODELS_PER_ROW = 8  # Modelos que se muestran por fila

FRAME_W = 640  # Altura y anchua de la camara
FRAME_H = 480


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class ColorParams:  # Clase con los marametros
    region: ROI
    patrones: list
    actual_method: str
    methods_list: list
    _index = 0

    def __init__(self, roi=None):
        if roi is None:
            raise Exception('ROI no valido')
        self.region = roi
        self.patrones = list()
        self.methods_list = list()

    def get_next_method(self):  # De haber más de una forma de comparar, seleciona la siguiente
        self._index = (self._index + 1) % len(self.methods_list)
        self.actual_method = self.methods_list[self._index]
        return self.actual_method


class Pattern:  # Clase que representa la información que almacena un patrón
    frame: np.ndarray
    color_info: tuple  # Datos del histograma de color

    def __init__(self, frame, color_info):
        self.frame = frame
        self.color_info = color_info


class Method:  # Representa un metodo de comparación
    fun = None
    selector: str

    def __init__(self, fun, select):
        self.fun = fun
        self.selector = select


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def bgr2gray(x):
    return cv.cvtColor(x, cv.COLOR_BGR2GRAY).astype(float) / 255


def gray2bgr(x):
    return cv.cvtColor(x, cv.COLOR_GRAY2BGR).astype(float) / 255


def make_histogram(c, size):  # Crea un histograma normalizado y el adaptado al tamaño del ROI
    h, b = np.histogram(c, np.arange(0, 257, 4))
    x = 2 * b[1:]
    yn = h / np.sum(h)
    y = size - h * (size / h.max())
    xy = np.array([x, y]).T.astype(int)
    xyn = np.array([x, yn])
    return xy, xyn


def make_rgb_histogram(f, size):  # Hace los histogramas para cada canal
    blue, green, red = cv.split(f)
    blue = make_histogram(blue, size)
    green = make_histogram(green, size)
    red = make_histogram(red, size)
    return blue, green, red


def resize(f, dim):  # Cambia el tamaño de una imagen
    return cv.resize(f, dim, interpolation=cv.INTER_LINEAR)


def stack_patterns(data, n=MODELS_PER_ROW):  # Funcion auxiliar para mostrar los histogramas
    result = list()
    i = 0
    aux_row = None
    for f in data.patrones:
        if i <= 0:
            if aux_row:
                result.append(np.hstack(aux_row))
            aux_row = list()
            aux_row.append(f.frame)
        else:
            aux_row.append(f.frame)
        i = (i + 1) % MODELS_PER_ROW

    if aux_row:
        while len(aux_row) < MODELS_PER_ROW and len(result) > 0:
            aux_row.append(np.zeros_like(f))

        result.append(np.hstack(aux_row))

    return np.vstack(result)


def hg_diff(hg1, hg2):  # Comparacion de histogramas por diferencia absoluta
    hgy1 = hg1[:, 1]
    hgy2 = hg2[:, 1]
    diff = np.abs(hgy1 - hgy2)
    return diff


def hg_intersect(hg1, hg2):  # Comparacion de histogramas por interseccion de histogramas
    mini = np.minimum(hg1, hg2)
    result = np.true_divide(np.sum(mini), np.sum(hg2))
    return result


def select_candidate(values, mode):  # Seleciona que canal se diferencia más
    if mode == 'min':
        return np.max(values)
    elif mode == 'max':
        return np.min(values)


def is_better(value, last_value, mode):  # Comprueba si el modelo actual es mejor
    if last_value is None:
        return True
    elif mode == 'min':
        return value < last_value
    elif mode == 'max':
        return value > last_value


def select_most_like_model(data, hgn, method):  # Selecciona el mejor modelo
    values = list()
    index = 0
    last_min = None
    b, g, r = hgn
    for i in range(0, len(data.patrones)):
        bp, gp, rp = data.patrones[i].color_info
        aux_b = method.fun(bp, b)
        aux_g = method.fun(gp, g)
        aux_r = method.fun(rp, r)
        value = select_candidate([aux_b, aux_g, aux_r], method.selector)
        values.append(value)
        if is_better(value, last_min, method.selector):
            index = i
            last_min = value

    return values, data.patrones[index]


def show_values(vals, img, n=MODELS_PER_ROW):  # Muestra los valores del analisis
    i = 0
    l = 1
    aux_row = ''
    for v in vals:

        if i == 0 and l == 1 and aux_row != '':
            putText(img, aux_row, orig=(5, 16 * l))
            aux_row = ''
            l += 1
        aux_row += f'{np.round(v, 3)}    '
        i = (i + 1) % n

    if aux_row != '':
        putText(img, aux_row, orig=(5, 16 * l))


# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

cv.namedWindow('input')
data = ColorParams(ROI('input'))
methods = dict()

methods['diferencia'] = Method(hg_diff, select='min')
methods['interseccion'] = Method(hg_intersect, select='max')
data.methods_list.append('diferencia')
data.actual_method = 'diferencia'
data.methods_list.append('interseccion')
# ---------------------------------------------------------------------------
# CODE
# ---------------------------------------------------------------------------

for key, frame in autoStream():
    if data.region.roi:
        [x1, y1, x2, y2] = data.region.roi
        recorte = frame[y1:y2, x1:x2]
        b, g, r = make_rgb_histogram(recorte, y2 - y1)

        if key == ord('c'):  # Guarda la region como modelo
            info = (b[1], g[1], r[1])
            data.patrones.append(Pattern(resize(recorte, MODEL_DIM), info))

        if key == ord('x'):  # Limpia la region
            data.region.roi = None
            continue

        if key == ord('r'):  # Borra todos los modelos
            data.patrones.clear()
            try:
                cv.destroyWindow('modelos')
            except Exception:
                pass

        if key == ord('n'):  # Salta al siguiete metodo
            data.get_next_method()

        if len(data.patrones) > 0:
            m = methods[data.actual_method]
            vals, model = select_most_like_model(data, (b[1], g[1], r[1]), m)
            show_values(vals, frame)
            cv.imshow('detectado', model.frame)

        cv.polylines(recorte, [b[0]], isClosed=False, color=(255, 0, 0),
                     thickness=2)  # Dibuja las lineas de visualización
        cv.polylines(recorte, [g[0]], isClosed=False, color=(0, 255, 0), thickness=2)
        cv.polylines(recorte, [r[0]], isClosed=False, color=(0, 0, 255), thickness=2)

        cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
        putText(frame, f'{x2 - x1 + 1}x{y2 - y1 + 1}', orig=(x1, y1 - 8))

        if len(data.patrones) > 0:
            cv.imshow('modelos', stack_patterns(data))

    putText(frame, f'{data.actual_method}', orig=(5, FRAME_H - 16))
    cv.imshow('input', frame)

cv.destroyAllWindows()
