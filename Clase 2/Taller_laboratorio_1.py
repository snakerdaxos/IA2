# ------------------------------------------------------------
# TALLER LABORATORIO 1 - Conversion a escala de grises
# ------------------------------------------------------------
# 1. Pixel amarillo intenso en formato BGR (Azul=0, Verde=255, Rojo=255)

import cv2
import numpy as np
pixel = np.array([0, 255, 255], dtype=np.uint8)

# 2. Valor de gris con la formula ponderada (producto punto, orden BGR)
W = np.array([0.114, 0.587, 0.299])                  # pesos [B, G, R]
Y = np.dot(pixel.astype(np.float64), W)               # 0.114*0 + 0.587*255 + 0.299*255
print(f'Valor matematico exacto: {Y}')                # -> 225.93
print(f'Valor de gris final (int): {int(round(Y))}')  # -> 226

# 4. Verificacion con OpenCV sobre una imagen real
imagen_real = cv2.imread('foto_demo.jpg')
img_gris = cv2.cvtColor(imagen_real, cv2.COLOR_BGR2GRAY)
print('Verificacion cv2.cvtColor OK, shape:', img_gris.shape)
