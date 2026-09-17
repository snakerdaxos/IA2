import numpy as np

# 1 Y 2. DEFINICIÓN DE LA NEURONA (PERCEPTRÓN DESDE CERO)
# Definir la Función de Activación (Escalón)[cite: 3]
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

# Definir la Estructura de la Neurona[cite: 3]
def perceptron(X, W, b):
    # Producto punto (Combinación lineal) + Sesgo
    Z = np.dot(X, W) + b
    # Activación
    salida = funcion_escalon(Z)
    return salida


# 3 Y 4. EL RETO: CONFIGURAR LOS PESOS PARA LA COMPUERTA OR
# Reglas de la compuerta OR:
# - [1, 1] -> 1
# - [1, 0] -> 1
# - [0, 1] -> 1
# - [0, 0] -> 0

# Modificamos MANUALMENTE los pesos (W) y el sesgo (b) para resolver la compuerta OR:
pesos_or = np.array([1.0, 1.0])  # W vector
sesgo_or = -0.5                  # b constante


# 5. VERIFICACIÓN Y PRUEBAS CON TODAS LAS ENTRADAS
print("--- PRUEBAS DE LA COMPUERTA OR ---")

# Caso 1: Entrada [1, 1]
resultado_1 = perceptron(np.array([1, 1]), pesos_or, sesgo_or)
print(f"Entrada [1, 1] -> Resultado: {resultado_1} (Esperado: 1)")

# Caso 2: Entrada [1, 0]
resultado_2 = perceptron(np.array([1, 0]), pesos_or, sesgo_or)
print(f"Entrada [1, 0] -> Resultado: {resultado_2} (Esperado: 1)")

# Caso 3: Entrada [0, 1]
resultado_3 = perceptron(np.array([0, 1]), pesos_or, sesgo_or)
print(f"Entrada [0, 1] -> Resultado: {resultado_3} (Esperado: 1)")

# Caso 4: Entrada [0, 0]
resultado_4 = perceptron(np.array([0, 0]), pesos_or, sesgo_or)
print(f"Entrada [0, 0] -> Resultado: {resultado_4} (Esperado: 0)")



# RESPUESTAS PARA EL INFORME / COMENTARIOS DE ANÁLISIS
# COMENTARIO DE RESPUESTA (Puntos 3 y 5):
# ¿Cuáles fueron los pesos y el sesgo que lograron resolver el problema?
# - Vector de pesos (W): [1.0, 1.0]
# - Sesgo (b): -0.5
# 
# Explicación matemática de por qué funciona:
# - Si la entrada es [0, 0]: Z = (0*1) + (0*1) - 0.5 = -0.5 -> Como Z < 0, la función escalón da 0.
# - Si la entrada es [1, 0]: Z = (1*1) + (0*1) - 0.5 = +0.5 -> Como Z >= 0, la función escalón da 1.
# - Si la entrada es [0, 1]: Z = (0*1) + (1*1) - 0.5 = +0.5 -> Como Z >= 0, la función escalón da 1.
# - Si la entrada es [1, 1]: Z = (1*1) + (1*1) - 0.5 = +1.5 -> Como Z >= 0, la función escalón da 1.
# 
# Con esto demostramos que ajustando manualmente los parámetros (pesos y sesgo) logramos 
# que el perceptrón emule la lógica de una compuerta OR, simulando el proceso de "entrenamiento" 
# donde las redes neuronales ajustan sus pesos automáticamente.
