from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
CARPETA_RESULTADOS.mkdir(exist_ok=True)

MOSTRAR_VENTANAS = False
UMBRAL_AREA_GRANDE = 4500   # lógica empresarial: objetos con más de 3000 px²

rng = np.random.default_rng(21)


# ---------------------------------------------------------------------------
# Paso 1: escena de "monedas" y una llave sobre fondo uniforme
# ---------------------------------------------------------------------------
def crear_monedas(alto=420, ancho=620):
    """Monedas = círculos rellenos con borde; la llave = círculo + rectángulo.
    Tonos más oscuros que el fondo para que la umbralización las aísle."""
    gris = np.full((alto, ancho), 205, np.uint8)   # fondo claro uniforme

    monedas = [
        (110, 130, 55),   # grande
        (300, 105, 38),   # mediana
        (500, 150, 62),   # grande
        (150, 320, 27),   # pequeña
        (330, 315, 45),   # mediana
        (520, 340, 29),   # pequeña
    ]
    for x, y, r in monedas:
        cv2.circle(gris, (x, y), r, 150, -1)       # relleno
        cv2.circle(gris, (x, y), r, 120, 4)        # borde (relieve)

    # Llave: cabeza circular + tallo rectangular (un solo objeto)
    cv2.circle(gris, (90, 90), 26, 150, -1)
    cv2.rectangle(gris, (90, 78), (240, 102), 150, -1)
    cv2.rectangle(gris, (218, 102), (240, 130), 150, -1)

    # Suavizado leve + ruido suave: como una foto real
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    gris = gris.astype(np.int16) + rng.integers(-4, 5, gris.shape)
    return np.clip(gris, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
def main():
    gris = crear_monedas()

    # -- Paso 2: pipeline de las 6 clases ------------------------------------
    # a) Ya está en grises. b) Umbralización de Otsu INVERTIDA:
    #    las monedas son MÁS OSCURAS que el fondo -> THRESH_BINARY_INV
    #    las convierte en blanco (objeto) sobre negro (fondo).
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # c) Limpieza morfológica (apertura) para quitar motas del fondo
    kernel = np.ones((3, 3), np.uint8)
    limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    # d) Detección de contornos (solo contornos externos: ignorar agujeros)
    contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # e) Métricas y dibujo sobre una versión a color
    lienzo = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

    print("=" * 64)
    print("CLASIFICADOR DE FORMAS — {} objetos detectados".format(len(contornos)))
    print("=" * 64)

    objetos = []
    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        x, y, w, h = cv2.boundingRect(contorno)

        # Momentos -> centroide
        m = cv2.moments(contorno)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        # Paso 4: lógica empresarial por área
        if area > UMBRAL_AREA_GRANDE:
            color, clase = (255, 0, 0), "GRANDE"
        else:
            color, clase = (0, 0, 255), "pequeño"

        cv2.rectangle(lienzo, (x, y), (x + w, y + h), color, 3)
        cv2.circle(lienzo, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(lienzo, "{:.0f}px".format(area), (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        objetos.append((area, clase, x, y, w, h, cx, cy))
        print("  Objeto {:>2}: area = {:>8.0f} px²  bbox=({:>3},{:>3},{:>3}x{:<3}) "
              "centroide=({:>3},{:>3})  -> {}".format(
                  len(objetos), area, x, y, w, h, cx, cy, clase))

    grandes = sum(1 for o in objetos if o[1] == "GRANDE")
    print("\nResumen: {} objetos GRANDES (bbox azul) y {} pequeños (bbox rojo)".format(
        grandes, len(objetos) - grandes))

    # -- Visualización ---------------------------------------------------------
    if MOSTRAR_VENTANAS:
        cv2.imshow("Clasificador de formas", lienzo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.8))
    titulos = [
        (gris, "1. Original (monedas + llave)"),
        (limpia, "2. Pipeline: Otsu + apertura\n(objetos blancos, fondo negro)"),
        (cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB),
         "3. Contornos: bbox AZUL > {}px², ROJO menor\n(y centroide en rojo)".format(UMBRAL_AREA_GRANDE)),
    ]
    for eje, (img, titulo) in zip(ejes, titulos):
        if img.ndim == 2:
            eje.imshow(img, cmap="gray")
        else:
            eje.imshow(img)
        eje.set_title(titulo, fontsize=11)
        eje.set_xticks([]), eje.set_yticks([])
    fig.suptitle("Laboratorio Clase 6 — Clasificador de formas (proyecto integrador)", fontsize=13)
    fig.tight_layout()
    ruta = CARPETA_RESULTADOS / "clasificacion_formas.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print("[FIGURA] Guardada:", ruta)

    print("""
CONCLUSIÓN
  Con las herramientas de las 6 clases ya se puede clasificar objetos por
  tamaño SIN redes neuronales: segmentar (Otsu) -> limpiar (apertura) ->
  contornear (findContours) -> medir (área, bbox, momentos) -> decidir
  con una regla de negocio (umbral de área). Es la base clásica sobre la
  que después se apoyan los detectores neuronales como el del proyecto.""")


if __name__ == "__main__":
    main()
