from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados"
CARPETA_RESULTADOS.mkdir(exist_ok=True)

MOSTRAR_VENTANAS = False
TAMANO_KERNEL = 7          

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Paso 1: escena sintética + ruido de sal y pimienta
# ---------------------------------------------------------------------------
def crear_escena(alto=420, ancho=640):
    """Ciudad nocturna simple: cielo con gradiente, edificios, ventanas,
    luna y una zona con textura (para ver qué pasa con los 'bordes falsos')."""
    # Cielo: gradiente vertical de claro (arriba) a oscuro (abajo)
    cielo = np.tile(np.linspace(150, 95, alto).reshape(-1, 1), (1, ancho))
    escena = cielo.astype(np.float32)

    # Edificios (rectángulos oscuros sobre el cielo)
    edificios = [(50, 200, 170, alto), (200, 140, 320, alto), (350, 250, 470, alto),
                 (490, 180, 620, alto)]
    for x1, y1, x2, y2 in edificios:
        cv2.rectangle(escena, (x1, y1), (x2, y2), 62, -1)

    # Ventanas (cuadritos claros regulares)
    for x1, y1, x2, y2 in edificios:
        for y in range(y1 + 18, y2 - 12, 26):
            for x in range(x1 + 14, x2 - 12, 30):
                if rng.random() < 0.55:
                    cv2.rectangle(escena, (x, y), (x + 12, y + 12), 205, -1)

    # Luna
    cv2.circle(escena, (565, 70), 34, 235, -1)

    # Zona de textura compleja (franjas diagonales finas en un edificio)
    for i, x in enumerate(range(355, 468, 7)):
        cv2.line(escena, (x, 258), (x + 40, alto), 95 if i % 2 else 75, 2)

    return np.clip(escena, 0, 255).astype(np.uint8)


def agregar_ruido_sal_pimienta(imagen, prob=0.025):
    """Inyecta píxeles completamente blancos (sal) y negros (pimienta)."""
    ruidosa = imagen.copy()
    total = imagen.size
    sal = rng.random(total) < prob            # sal: blanco
    pimienta = rng.random(total) < prob       # pimienta: negro
    ruidosa.ravel()[sal] = 255
    ruidosa.ravel()[pimienta] = 0
    return ruidosa


# ---------------------------------------------------------------------------
# Métricas para el análisis crítico (punto 4)
# ---------------------------------------------------------------------------
def contar_pixeles_extremos(imagen):
    """Cuántos píxeles siguen siendo 0 o 255 puros."""
    return int(((imagen == 0) | (imagen == 255)).sum())


def error_mae(imagen_a, imagen_b):
    """Error absoluto medio contra la referencia (imagen limpia)."""
    diferencia = np.abs(imagen_a.astype(np.float32) - imagen_b.astype(np.float32))
    return float(diferencia.mean())


# ---------------------------------------------------------------------------
def main():
    limpia = crear_escena()
    ruidosa = agregar_ruido_sal_pimienta(limpia)

    # Paso 2: los tres filtros con kernel agresivo 7x7
    filtrada_media = cv2.blur(ruidosa, (TAMANO_KERNEL, TAMANO_KERNEL))
    filtrada_gauss = cv2.GaussianBlur(ruidosa, (TAMANO_KERNEL, TAMANO_KERNEL), 0)
    filtrada_mediana = cv2.medianBlur(ruidosa, TAMANO_KERNEL)

    # Paso 3: visualización
    if MOSTRAR_VENTANAS:
        for nombre, img in [("Con ruido", ruidosa),
                            ("Media 7x7", filtrada_media),
                            ("Gaussiano 7x7", filtrada_gauss),
                            ("Mediana 7x7", filtrada_mediana)]:
            cv2.imshow(nombre, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -- Panel completo -------------------------------------------------------
    fig, ejes = plt.subplots(2, 2, figsize=(12.5, 8))
    paneles = [
        (ruidosa, "1. Original con ruido sal y pimienta (5%)"),
        (filtrada_media, "2. MEDIA 7x7 (aparecen 'manchas' grises)"),
        (filtrada_gauss, "3. GAUSSIANO 7x7 (manchas más suaves)"),
        (filtrada_mediana, "4. MEDIANA 7x7 (ruido eliminado limpio)"),
    ]
    for eje, (img, titulo) in zip(ejes.flat, paneles):
        eje.imshow(img, cmap="gray", vmin=0, vmax=255)
        eje.set_title(titulo, fontsize=11)
        eje.set_xticks([]), eje.set_yticks([])
    fig.suptitle("Laboratorio Clase 4 — Estrategias de suavizado (kernel {}x{})".format(
        TAMANO_KERNEL, TAMANO_KERNEL), fontsize=13)
    fig.tight_layout()
    ruta = CARPETA_RESULTADOS / "comparacion_filtros.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print("[FIGURA] Guardada:", ruta)

    # -- Zoom sobre una zona con ruido ---------------------------------------
    cy, cx = 90, 300            # zona del cielo con puntos de ruido
    lado = 45
    recorte = (slice(cy - lado, cy + lado), slice(cx - lado, cx + lado))

    fig, ejes = plt.subplots(1, 4, figsize=(14, 3.8))
    for eje, (img, titulo) in zip(ejes, paneles):
        eje.imshow(img[recorte], cmap="gray", vmin=0, vmax=255,
                   interpolation="nearest")
        eje.set_title(titulo.split(". ")[1], fontsize=10)
        eje.set_xticks([]), eje.set_yticks([])
    fig.suptitle("Zoom x1 (45x45 px): cómo queda cada píxel de ruido")
    fig.tight_layout()
    ruta_zoom = CARPETA_RESULTADOS / "zoom_filtros.png"
    fig.savefig(ruta_zoom, dpi=150)
    plt.close(fig)
    print("[FIGURA] Guardada:", ruta_zoom)

    # -- Paso 4: análisis crítico con métricas --------------------------------
    print("\nANÁLISIS CRÍTICO (punto 4)")
    print("{:<22} {:>18} {:>18}".format("Imagen", "Píxeles 0/255", "MAE vs limpia"))
    filas = [
        ("Con ruido", ruidosa),
        ("Media 7x7", filtrada_media),
        ("Gaussiano 7x7", filtrada_gauss),
        ("Mediana 7x7", filtrada_mediana),
    ]
    for nombre, img in filas:
        print("{:<22} {:>18} {:>18.3f}".format(
            nombre, contar_pixeles_extremos(img), error_mae(img, limpia)))

    print("\n¿Por qué la MEDIANA ignora los extremos?")
    print("  Ordena los {} valores bajo el kernel y elige el del medio (posición".format(TAMANO_KERNEL ** 2))
    print("  25). Los 0 y 255 del ruido quedan en los extremos de la lista")
    print("  ordenada y NUNCA resultan elegidos.")
    print("¿Por qué la MEDIA crea manchas grises?")
    print("  Promedia los {} valores incluyendo el ruido: un 255 promediado con".format(TAMANO_KERNEL ** 2))
    print("  48 píxeles oscuros deja un halo de ~5 niveles por encima del fondo,")
    print("  y además arrastra (suaviza) los bordes reales de los edificios.")
    print("\nCONCLUSIÓN: para ruido de impulso (sal y pimienta) el filtro de")
    print("MEDIANA es el correcto; media y gaussiano sirven para ruido gaussiano")
    print("leve, no para valores extremos aislados.")


if __name__ == "__main__":
    main()

