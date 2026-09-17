import numpy as np
 
# 1. Matriz de prueba 5x5 con valores aleatorios entre 200 y 254
np.random.seed(42)                      # opcional: hace el ejemplo reproducible
A = np.random.randint(200, 255, (5, 5))
 
# 2. Parametros: reduccion de contraste 50% y brillo -50
alpha = 0.5
beta = -50.0
 
# 3. Transformacion lineal + clipping + conversion a enteros 8 bits
A_procesada = alpha * A.astype(np.float32) + beta
A_procesada = np.clip(A_procesada, 0, 255).astype(np.uint8)
 
# 4. Comparacion
print('Matriz original:')
print(A)
print('Matriz procesada (alpha=0.5, beta=-50):')
print(A_procesada)
