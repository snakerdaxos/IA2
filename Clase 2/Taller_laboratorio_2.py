# ------------------------------------------------------------
# TALLER LABORATORIO 2 - Analisis estadistico (histogramas)
# ------------------------------------------------------------
# 1. Cargar imagen RGB
imagen = cv2.imread('foto_demo.jpg')

# 2. Separar en 3 canales (OpenCV usa orden BGR)
canal_b = imagen[:, :, 0]
canal_g = imagen[:, :, 1]
canal_r = imagen[:, :, 2]

# 3. Histograma de cada canal
hist_b = cv2.calcHist([canal_b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([canal_g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([canal_r], [0], None, [256], [0, 256])

# 4. Graficar los tres histogramas superpuestos
plt.figure(figsize=(10, 5))
plt.plot(hist_b, color='blue',   label='Canal Azul')
plt.plot(hist_g, color='green',  label='Canal Verde')
plt.plot(hist_r, color='red',    label='Canal Rojo')
plt.title('Distribucion de Intensidades por Canal')
plt.xlabel('Valor del Pixel (0-255)')
plt.ylabel('Frecuencia (Cantidad de pixeles)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('histograma_demo.png', dpi=110, bbox_inches='tight')
plt.show()

# 5. Conclusion: color dominante = canal con mayor media
medias = {'AZUL': canal_b.mean(), 'VERDE': canal_g.mean(), 'ROJO': canal_r.mean()}
for color, media in medias.items():
    print(f'Media {color}: {media:.1f}')
dominante = max(medias, key=medias.get)
print(f'CONCLUSION: el color dominante en la iluminacion es el {dominante}')