import os
import numpy as np
import cv2

# 1. Cargar imagen real o generar una de prueba reproducible
imagen = cv2.imread('documento.jpg', cv2.IMREAD_GRAYSCALE)
if imagen is None:
    print("No se encontro 'documento.jpg' -> generando imagen de prueba...")
    imagen = np.full((120, 160), 90, np.uint8)        # fondo gris oscuro
    cv2.circle(imagen, (80, 60), 35, 180, -1)         # objeto claro
    for (x, y) in [(55, 70), (85, 60), (90, 75)]:     # huecos internos
        cv2.circle(imagen, (x, y), 4, 60, -1)
    for i in np.random.default_rng(7).integers(0, imagen.size, 10):
        imagen.flat[i] = 200                          # ruido sal en el fondo

# 2. Umbral estatico que deja entrar ruido intencional
_, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)

# 3. Elemento Estructurante de 3x3
kernel = np.ones((3, 3), np.uint8)

# 4. Apertura (Erosion + Dilatacion): limpia el fondo
apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

# 5. Cierre (Dilatacion + Erosion): rellena huecos internos
cierre = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

# 6. Resultados y conclusion (en clase usar cv2.imshow para las 3 ventanas)
os.makedirs('resultados', exist_ok=True)
for nombre, img in [('1_grises.png', imagen), ('2_binaria.png', binaria),
                    ('3_apertura.png', apertura), ('4_cierre.png', cierre)]:
    cv2.imwrite(os.path.join('resultados', nombre), img)

blancos = lambda m: int((m == 255).sum())
print(f"Pixels blancos  binarizada: {blancos(binaria)}")
print(f"Pixels blancos  apertura:   {blancos(apertura)}  "
      f"(elimino {blancos(binaria) - blancos(apertura)} px de ruido de sal)")
print(f"Pixels blancos  cierre:     {blancos(cierre)}  "
      f"(relleno {blancos(cierre) - blancos(binaria)} px de huecos internos)")
print()
print("CONCLUSION: para esta imagen gano la APERTURA, porque el problema")
print("era ruido de sal (puntos blancos en el fondo). El cierre no quita")
print("ese ruido (solo rellena huecos del objeto). Regla general:")
print("  ruido de sal -> APERTURA | ruido de pimienta -> CIERRE")
print()
print("Imagenes guardadas en resultados/ (en clase: cv2.imshow x3 + waitKey)")
