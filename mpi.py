import numpy as np

def binarizar(imagen_gris, umbral=127):
    # Binario invertido: oscuro (pastilla) se vuelve 1 (blanco)
    return np.where(imagen_gris > umbral, 0, 1).astype(np.uint8)

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

def cerrar(binaria, kernel_size=5, iteraciones=2):
    kernel = np.ones((kernel_size, kernel_size))
    for _ in range(iteraciones):
        binaria = dilatar(binaria, kernel)
        binaria = erosionar(binaria, kernel)
    return binaria

def tiene_pastilla(parche_gray, umbral=127, kernel_size=5, iteraciones=2, umbral_negro=0.3):
    binaria = binarizar(parche_gray, umbral)
    cerrada = cerrar(binaria, kernel_size, iteraciones)
    proporcion_blanco = np.mean(cerrada)
    return proporcion_blanco > (1 - umbral_negro)
