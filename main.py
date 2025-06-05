# main.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from mpi import binarizar, cerrar, tiene_pastilla

st.title("📋 Detección Automática de Pastillas Faltantes")

st.write("""
Sube una imagen del blíster para detectar automáticamente cuántas pastillas hay presentes y cuántas faltan.
""")

uploaded_file = st.file_uploader("Sube una imagen del blíster", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = np.array(image)[:, :, ::-1] 
    st.image(image, caption="Imagen original", use_container_width=True)

    TARGET_WIDTH = 600
    scale_ratio = TARGET_WIDTH / img_cv.shape[1]
    new_dim = (TARGET_WIDTH, int(img_cv.shape[0] * scale_ratio))
    img_cv_resized = cv2.resize(img_cv, new_dim)

    gray = cv2.cvtColor(img_cv_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    dp = 1.2
    minDist = 35
    param1 = 30
    param2 = 25
    minRadius = 16
    maxRadius = 23

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

    salida = img_cv_resized.copy()
    total = 0
    vacias = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        total = circles.shape[1]

        for c in circles[0, :]:
            x, y, r = c
            x1 = max(x - r, 0)
            y1 = max(y - r, 0)
            x2 = min(x + r, gray.shape[1])
            y2 = min(y + r, gray.shape[0])
            parche = gray[y1:y2, x1:x2]

            if tiene_pastilla(parche):
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
                vacias += 1

            cv2.circle(salida, (x, y), r, color, 3)
            cv2.circle(salida, (x, y), 2, (255, 0, 0), 3)

        st.write(f"🔎 Total de posiciones detectadas: **{total}**")
        st.write(f"❌ Pastillas faltantes: **{vacias}**")

        if vacias > 0:
            st.error("⚠️ Se encontraron pastillas faltantes.")
        else:
            st.success("✅ Todas las pastillas están presentes.")

    else:
        st.warning("⚠️ No se detectaron círculos. Prueba con otra imagen o mejora la iluminación/contraste.")

    salida_rgb = cv2.cvtColor(salida, cv2.COLOR_BGR2RGB)
    st.image(salida_rgb, caption="Resultado", use_container_width=True)
