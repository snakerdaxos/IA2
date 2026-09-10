import numpy as np
 
# 1. Seccion de imagen y kernel de realce (tomados del diagrama)
I = np.array([[100, 100, 100],
              [100, 200, 100],
              [100, 100, 100]])
 
K = np.array([[ 0, -1,  0],
              [-1,  5, -1],
              [ 0, -1,  0]])
 
# 2. Superposicion: producto Hadamard (elemento a elemento)
producto = I * K
 
# 3. Suma de todos los valores -> pixel central resultante
pixel_central = np.sum(producto)
 
print('Producto Hadamard I*K:')
print(producto)
print('Valor del pixel central calculado:', pixel_central)
