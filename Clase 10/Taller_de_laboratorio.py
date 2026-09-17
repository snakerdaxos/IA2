import numpy as np
from sklearn.svm import SVC


# 1 Y 2. CREACIÓN Y MODIFICACIÓN DEL DATASET
# Agregamos el nuevo punto [5, 5] y le asignamos la etiqueta 0 (Clase A) 
# para "engañar" la frontera lineal (este punto queda metido en el territorio de la Clase B).
X = np.array([
    [2, 2],
    [3, 3],
    [4, 2],
    [6, 6],
    [7, 8],
    [8, 7],
    [5, 5]  # Nuevo punto añadido (Clase A / 0)
])

Y = np.array([0, 0, 0, 1, 1, 1, 0])


# 3. ENTRENAMIENTO CON KERNEL LINEAL
# Inicializamos SVM con Kernel Lineal (intenta trazar una recta rígida)[cite: 2]
modelo_svm_lineal = SVC(kernel='linear')
modelo_svm_lineal.fit(X, Y)

# Extraemos los vectores de soporte
vectores_lineal = modelo_svm_lineal.support_vectors_
print("Vectores de soporte (Kernel Lineal):\n", vectores_lineal)

# Predicción con modelo lineal
nuevo_punto = np.array([[5, 4]])
pred_lineal = modelo_svm_lineal.predict(nuevo_punto)
print("Predicción para [5,4] con Kernel Lineal:", pred_lineal[0])

# COMENTARIO DE ANÁLISIS (Puntos 1 y 3):
# ¿Coinciden con los del taller analítico? 
# Al introducir el punto [5, 5] de la Clase A en medio del grupo B, el modelo lineal 
# se ve forzado o falla, ya que es matemáticamente imposible trazar una línea recta perfecta 
# que separe ambos grupos sin cruzar datos o destruir el margen de seguridad.



# 4. EXPERIMENTACIÓN CON KERNEL RBF 
# Cambiamos el hiperparámetro de kernel='linear' a kernel='rbf' (Radial Basis Function)[cite: 2]
modelo_svm_rbf = SVC(kernel='rbf')
modelo_svm_rbf.fit(X, Y)

# Predicción con modelo RBF para el mismo punto nuevo
pred_rbf = modelo_svm_rbf.predict(nuevo_punto)
print("Predicción para [5,4] con Kernel RBF:", pred_rbf[0])



# 5. REFLEXIÓN: KERNEL TRICK (RBF) EN EL MUNDO REAL
# COMENTARIO DE RESPUESTA:
# ¿En qué escenario del mundo real un kernel lineal fallaría y se requeriría RBF?
# Escenario: MEDICINA (Clasificación de células tumorales o tejidos).
# Explicación: En diagnósticos médicos por imágenes o biopsias, a menudo las células sanas 
# forman un núcleo o un anillo que está completamente rodeado por células malignas (o viceversa). 
# Una línea recta o un plano plano jamás podría separar estos grupos ("el círculo dentro de otro grupo"). 
# El Kernel RBF utiliza el "Kernel Trick" para proyectar los datos a una dimensión superior 
# (como levantar el centro con un embudo en 3D), permitiendo trazar fronteras curvas capaces 
# de aislar el problema con total precisión.
