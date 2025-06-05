import numpy as np
import cv2

# Convierte imagen a binaria con umbral invertido
def binarizar(imagen_gris, umbral=127):
    _, binarizada = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY_INV)
    return binarizada

# Aplica cierre morfológico a una imagen binaria
def cerrar(imagen_binaria, kernel_size=5, iteraciones=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)

# Determina si una región contiene una pastilla con base en proporción de pixeles blancos
# umbral_negro define cuánta proporción blanca se acepta antes de considerar que no hay pastilla
def tiene_pastilla(parche_gray, umbral=127, kernel_size=5, iteraciones=2, umbral_negro=0.3):
    parche_bin = binarizar(parche_gray, umbral)
    parche_cerrado = cerrar(parche_bin, kernel_size, iteraciones)
    black_ratio = np.sum(parche_cerrado == 255) / parche_cerrado.size
    return black_ratio < umbral_negro
