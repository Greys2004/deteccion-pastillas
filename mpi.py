# mpi.py
import numpy as np

# -----------------------------
# TUS FUNCIONES AUXILIARES
# -----------------------------

def binarizar(imagen_gris, umbral=100):
    """
    Convierte la imagen en escala de grises a binaria usando el umbral dado.
    """
    return np.where(imagen_gris > umbral, 1.0, 0.0)

def dilatar(binaria, kernel):
    h, w = binaria.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    salida = np.zeros_like(binaria)

    for y in range(ph, h - ph):
        for x in range(pw, w - pw):
            region = binaria[y - ph:y + ph + 1, x - pw:x + pw + 1]
            salida[y, x] = np.max(region * kernel)

    return salida

def erosionar(binaria, kernel):
    h, w = binaria.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    salida = np.zeros_like(binaria)

    for y in range(ph, h - ph):
        for x in range(pw, w - pw):
            region = binaria[y - ph:y + ph + 1, x - pw:x + pw + 1]
            salida[y, x] = np.min(region * kernel)

    return salida

def cerrar(binaria, kernel=np.ones((5, 5))):
    return erosionar(dilatar(binaria, kernel), kernel)

# ----------------------------------
# FUNCIÓN FINAL QUE USA TODO ESO
# ----------------------------------

def tiene_pastilla(parche_gris):
    """
    Detecta si hay una pastilla presente en un parche de imagen gris.
    """
    binaria = binarizar(parche_gris)
    cerrada = cerrar(binaria)
    porcentaje_blanco = np.mean(cerrada)
    return porcentaje_blanco > 0.2 
