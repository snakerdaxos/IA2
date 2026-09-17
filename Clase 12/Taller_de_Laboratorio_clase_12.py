import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))


# 3. EL RETO DIMENSIONAL: PROCESAMIENTO EN LOTE (BATCH)
# Modificamos X para que sea una matriz de 2x3 (2 clientes, 3 características cada uno):
# Cliente 1: [0.5, 0.8, 0.2]
# Cliente 2: [0.1, 0.9, 0.9]
X = np.array([
    [0.5, 0.8, 0.2],
    [0.1, 0.9, 0.9]
])


# 2. CAPA OCULTA (4 Neuronas)
# Matriz W1 de (3 entradas x 4 neuronas)[cite: 4]
W1 = np.array([
    [ 0.1,  0.2,  0.3,  0.4],
    [-0.5,  0.6,  0.7, -0.8],
    [ 0.9, -0.1,  0.2,  0.3]
])

b1 = np.array([0.1, 0.2, 0.3, 0.4])  # 4 Sesgos[cite: 4]

# PROCESO CAPA OCULTA (Multiplicación matricial automática para todo el lote)
Z1 = np.dot(X, W1) + b1
A1 = sigmoide(Z1)  # Salida de la capa oculta (transformada al rango 0 y 1)


# 3. CAPA DE SALIDA (1 Neurona)
# Matriz W2 de (4 entradas ocultas x 1 neurona final)[cite: 4]
W2 = np.array([0.5, 0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# PROCESO CAPA FINAL
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)


# 1 Y 2. IMPRESIÓN Y ANÁLISIS DE RESULTADOS
print("--- VALORES INTERMEDIOS ---")
print("Z1 (Valores puros de la capa oculta):\n", Z1)
print("\nA1 (Valores de Z1 tras pasar por la Sigmoide):\n", A1)

print("\n--- PREDICCIÓN FINAL EN LOTE ---")
print("Predicciones de la Red (Probabilidades para los 2 clientes):", np.round(Salida_Final, 4))



# RESPUESTAS PARA EL INFORME / COMENTARIOS DE ANÁLISIS

# COMENTARIO DE RESPUESTA (Puntos 2 y 4):
# 1. ¿Cómo transformó la función sigmoide los valores puros de Z1 al rango (0, 1)?
#    La función sigmoide actúa como una función de activación no lineal que toma cualquier 
#    número real (positivo o negativo, grande o pequeño) contenido en la matriz Z1 y lo 
#    comprime o acota matemáticamente en un rango estricto entre 0 y 1. Esto permite interpretar 
#    los resultados intermedios como niveles de activación o probabilidades.
#
# 2. ¿Qué ventaja ofrece el procesamiento en lote (Batch) con `np.dot`?
#    Gracias a las propiedades del álgebra lineal y el cálculo tensorial, al pasar X como una 
#    matriz de 2x3 en lugar de un vector unidimensional, `np.dot` multiplica simultáneamente 
#    todas las filas de los clientes contra la matriz de pesos W1 en una sola operación optimizada. 
#    Esto demuestra cómo las redes neuronales pueden procesar miles de datos en paralelo 
#    (aprovechando la arquitectura de las GPUs) sin necesidad de modificar las estructuras de pesos 
#    ni utilizar ciclos `for` tradicionales.
