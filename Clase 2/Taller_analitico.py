# ------------------------------------------------------------
# TALLER ANALITICO 1 
# ------------------------------------------------------------
imagen = np.zeros((1080, 1920, 3), dtype=np.uint8)   # foto 1920x1080
recorte = imagen[100:200, 300:400, 1]               # [filas, columnas, canal]
print('Shape de recorte:', recorte.shape)            # -> (100, 100)
