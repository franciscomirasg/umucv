import os

import cv2
import dlib

predictor_path = '/media/shadowgunther/Vault/Users/pacop/Documents/.git/umucv/data/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

import cv2 as cv
import numpy as np
from umucv.stream import autoStream


def extract_points(shape, range_p=range(36, 42, 1), complete=True):
    aux = []
    for p in range_p:
        x = shape.part(p).x
        y = shape.part(p).y
        aux.append(np.array([x, y]))

    if not complete:
        p = range_p[0]
        x = shape.part(p).x
        y = shape.part(p).y
        aux.append(np.array([x, y]))

    return np.array(aux)


def scale(f, scale):
    h = f.shape[0] * scale
    w = f.shape[1] * scale
    h = int(h)
    w = int(w)
    dim = (w, h)
    return cv.resize(f, dim, interpolation=cv2.INTER_AREA)


def make_mask(frame, contornos):
    mask = np.zeros_like(frame)
    cv.drawContours(mask, contornos, -1, (1, 1, 1), -1)
    return mask


for key, frame in autoStream():
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    dets = detector(img, 0)
    for k, d in enumerate(dets):
        cv.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 128, 64))

        shape = predictor(img, d)
        ojo_izq = extract_points(shape, range_p=range(36, 42))
        ojo_dch = extract_points(shape, range_p=range(42, 48))
        nariz = extract_points(shape, range_p=range(27, 36), complete=False)
        boca = extract_points(shape, range_p=range(48, 60))

        L = [ojo_izq, ojo_dch, nariz, boca]
        mask = make_mask(img, L)
        work = np.vstack([scale(mask, 0.5) * 255, scale(mask, 0.5) * scale(frame, 0.5)])
        work = np.hstack([frame, work])

    cv.imshow("face landmarks", work)
