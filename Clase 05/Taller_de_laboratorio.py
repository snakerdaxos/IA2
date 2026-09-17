from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
CARPETA_RESULTADOS.mkdir(exist_ok=True)

MOSTRAR_VENTANAS = False

rng = np.random.default_rng(11)


# ---------------------------------------------------------------------------
# Paso 1: escena con formas claras + textura compleja
# ---------------------------------------------------------------------------
def crear_escena(alto=380, ancho=560):
    """Formas geométricas nítidas + una zona de textura (ruido) que sirve
    de 'trampa' para los umbrales bajos de Canny."""
    imagen = np.full((alto, ancho), 110, np.uint8)   # fondo gris medio

    # Rectángulo claro (borde fuerte)
    cv2.rectangle(imagen, (40, 50), (200, 210), 220, -1)
    # Círculo gris claro (borde SUAVE: desaparece con umbrales altos)
    cv2.circle(imagen, (330, 130), 70, 165, -1)
    # Triángulo oscuro
    triangulo = np.array([[420, 210], [490, 80], [540, 210]], np.int32)
    cv2.fillPoly(imagen, [triangulo], 45)
    # Línea fina diagonal
    cv2.line(imagen, (60, 300), (500, 250), 240, 2)

    # Zona de textura compleja: ruido uniforme (como pasto o asfalto granulado)
    zona = (slice(250, 360), slice(40, 260))
    imagen[zona] = rng.integers(85, 140, size=(110, 220)).astype(np.uint8)

    return imagen


# ---------------------------------------------------------------------------
def main():
    imagen = crear_escena()

    # Paso 2: los tres detectores
    # Sobel X: derivada en dirección horizontal -> resalta bordes VERTICALES
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    bordes_verticales = cv2.convertScaleAbs(sobel_x)

    # Sobel Y: derivada en dirección vertical -> resalta bordes HORIZONTALES
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    bordes_horizontales = cv2.convertScaleAbs(sobel_y)

    # Canny con los umbrales estándar del taller (bajo=50, alto=150)
    bordes_canny = cv2.Canny(imagen, 50, 150)

    # Paso 3: panel de visualización
    if MOSTRAR_VENTANAS:
        for nombre, img in [("Sobel X (verticales)", bordes_verticales),
                            ("Sobel Y (horizontales)", bordes_horizontales),
                            ("Canny 50/150", bordes_canny)]:
            cv2.imshow(nombre, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    fig, ejes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    paneles = [
        (imagen, "1. Original (formas claras + zona de textura)"),
        (bordes_verticales, "2. SOBEL X: solo bordes VERTICALES"),
        (bordes_horizontales, "3. SOBEL Y: solo bordes HORIZONTALES"),
        (bordes_canny, "4. CANNY (50, 150): bordes finos de 1 píxel"),
    ]
    for eje, (img, titulo) in zip(ejes.flat, paneles):
        eje.imshow(img, cmap="gray")
        eje.set_title(titulo, fontsize=11)
        eje.set_xticks([]), eje.set_yticks([])
    fig.suptitle("Laboratorio Clase 5 — Inspector de bordes", fontsize=13)
    fig.tight_layout()
    ruta = CARPETA_RESULTADOS / "panel_bordes.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print("[FIGURA] Guardada:", ruta)

    # Paso 4: experimentación con los umbrales de Canny
    configuraciones = [(10, 50), (50, 150), (200, 250)]
    resultados_canny = {u: cv2.Canny(imagen, u[0], u[1]) for u in configuraciones}

    def cantidad_borde(img):
        return int((img > 0).sum())

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.6))
    for eje, (bajo, alto) in zip(ejes, configuraciones):
        img = resultados_canny[(bajo, alto)]
        eje.imshow(img, cmap="gray")
        eje.set_title("Canny({}, {})\n{} píxeles de borde".format(
            bajo, alto, cantidad_borde(img)), fontsize=11)
        eje.set_xticks([]), eje.set_yticks([])
    fig.suptitle("Experimento: los umbrales de histéresis lo cambian todo")
    fig.tight_layout()
    ruta_exp = CARPETA_RESULTADOS / "experimento_umbrales_canny.png"
    fig.savefig(ruta_exp, dpi=130)
    plt.close(fig)
    print("[FIGURA] Guardada:", ruta_exp)

    print("\nOBSERVACIONES DEL EXPERIMENTO")
    for (bajo, alto), img in resultados_canny.items():
        print("  Canny({}, {}): {:>6} píxeles de borde".format(
            bajo, alto, cantidad_borde(img)))
    print("""
  * Umbrales BAJOS (10/50): la zona de textura explota en bordes falsos
    ('fideos'): el ruido supera el umbral bajo y la histéresis conserva
    todo lo conectado a un borde fuerte. Bordes = ruido.
  * Umbrales ALTOS (200/250): sobreviven solo los cambios violentos;
    el círculo gris claro (borde suave real) DESAPARECE.
  * Óptimo empírico para ESTA imagen: 50/150 — detecta las fronteras
    reales de las formas e ignora la textura.
  * Moraleja: los umbrales no son universales; se calibran según el
    contraste y el ruido de cada imagen (igual que UMBRAL_RECONOCIMIENTO
    en el proyecto de reconocimiento facial).""")


if __name__ == "__main__":
    main()

