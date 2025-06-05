import numpy as np

# ------------------------
# Binarización invertida
# ------------------------
def binarizar(imagen_gris, umbral=127):
    return np.where(imagen_gris > umbral, 0, 255).astype(np.uint8)

# ------------------------
# Dilatación binaria
# ------------------------
def dilatar(imagen_binaria, kernel):
    h, w = imagen_binaria.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(imagen_binaria, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    salida = np.zeros_like(imagen_binaria)

    for y in range(h):
        for x in range(w):
            region = padded[y:y+kh, x:x+kw]
            salida[y, x] = 255 if np.any(region & kernel * 255) else 0

    return salida

# ------------------------
# Erosión binaria
# ------------------------
def erosionar(imagen_binaria, kernel):
    h, w = imagen_binaria.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(imagen_binaria, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    salida = np.zeros_like(imagen_binaria)

    for y in range(h):
        for x in range(w):
            region = padded[y:y+kh, x:x+kw]
            salida[y, x] = 255 if np.all(region & kernel * 255) else 0

    return salida

# ------------------------
# Cierre morfológico
# ------------------------
def cerrar(imagen_binaria, kernel_size=5, iteraciones=2):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    salida = imagen_binaria.copy()
    for _ in range(iteraciones):
        salida = dilatar(salida, kernel)
    for _ in range(iteraciones):
        salida = erosionar(salida, kernel)
    return salida

# ------------------------
# Verificar si hay pastilla
# ------------------------
def tiene_pastilla(parche_gray, umbral=127, kernel_size=5, iteraciones=2, umbral_negro=0.2):
    parche_bin = binarizar(parche_gray, umbral)
    parche_cerrado = cerrar(parche_bin, kernel_size, iteraciones)

    h, w = parche_cerrado.shape
    margin = int(min(h, w) * 0.2)
    centro = parche_cerrado[margin:h-margin, margin:w-margin]

    black_ratio = np.sum(centro == 255) / centro.size
    return black_ratio < umbral_negro
