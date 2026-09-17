import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# 1 Y 2. DATASET DE ENTRENAMIENTO AMPLIADO

# Se amplía el dataset a 10 puntos (filas) y 3 dimensiones (columnas):
# [Edad, Salario (en miles), Número de Hijos]
X_entrenamiento = np.array([
    [20, 30, 0],  # Punto 1
    [40, 50, 2],  # Punto 2
    [35, 45, 1],  # Punto 3
    [25, 20, 0],  # Punto 4
    [50, 80, 3],  # Punto 5
    [23, 25, 0],  # Punto 6
    [45, 60, 2],  # Punto 7
    [30, 35, 1],  # Punto 8
    [55, 90, 4],  # Punto 9
    [28, 28, 0]   # Punto 10
])


# 3. ETIQUETAS DE CLASIFICACIÓN

# Etiquetas correspondientes: 0 = NO COMPRA, 1 = COMPRA
Y_entrenamiento = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])

# Definimos el punto del nuevo cliente a evaluar: [Edad, Salario, Hijos]
nuevo_cliente = np.array([[30, 40, 1]])


# 4. EXPERIMENTACIÓN CON EL VALOR DE K


# --- Experimento A: K = 1 ---
# Instanciamos el modelo con n_neighbors = 1[cite: 1]
modelo_knn_1 = KNeighborsClassifier(n_neighbors=1)
modelo_knn_1.fit(X_entrenamiento, Y_entrenamiento) # Memoriza los datos
prediccion_1 = modelo_knn_1.predict(nuevo_cliente)
print("Clase predicha con K=1:", prediccion_1[0])
# COMENTARIO DE RESPUESTA (K=1):
# Con K=1, el algoritmo consulta únicamente al vecino más cercano en el espacio tridimensional.
# La decisión depende de un solo punto de referencia, lo que puede hacer que el modelo sea 
# sensible a valores atípicos (ruido).

# --- Experimento B: K = 5 ---
# Cambiamos a n_neighbors = 5[cite: 1]
modelo_knn_5 = KNeighborsClassifier(n_neighbors=5)
modelo_knn_5.fit(X_entrenamiento, Y_entrenamiento)
prediccion_5 = modelo_knn_5.predict(nuevo_cliente)
print("Clase predicha con K=5:", prediccion_5[0])
# COMENTARIO DE RESPUESTA (K=5):
# Con K=5, el algoritmo busca los 5 vecinos más cercanos y toma la decisión final mediante 
# votación democrática (mayoría). Esto aporta mayor estabilidad y robustez al modelo frente a 
# posibles excepciones o errores en el dataset.



# 5. PREGUNTA DE ANÁLISIS: LA MALDICIÓN DE LA DIMENSIONALIDAD

# COMENTARIO DE RESPUESTA:
# Si en lugar de 3 columnas tuviéramos 1,000 columnas (como los píxeles de una imagen):
# 1. Dispersión del espacio: El volumen del espacio multidimensional crece exponencialmente, 
#    provocando que los puntos de datos queden extremadamente dispersos y lejanos entre sí.
# 2. Pérdida de contraste: La distancia matemática (Euclidiana) entre el vecino más cercano 
#    y el más lejano tiende a volverse casi idéntica (las distancias relativas colapsan).
# 3. Impacto en KNN: Como KNN depende estrictamente de medir distancias para definir quiénes 
#    son los "vecinos", con 1,000 dimensiones todos los puntos parecerán estar equidistantemente 
#    lejos, destruyendo el concepto de vecindad y haciendo que el clasificador falle en sus 
#    predicciones. Por esta razón, se requiere reducir dimensiones (ej. con PCA) antes de usar KNN.
