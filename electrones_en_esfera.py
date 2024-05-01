import numpy as np

def f_electron(x):
    np = len(x) // 3  # Cada punto tiene tres coordenadas
    sum_repulsion = 0
    # Iterar sobre todos los pares de puntos
    for i in range(np):
        for j in range(i + 1, np):
            xi = x[3*i:3*i+3]
            xj = x[3*j:3*j+3]
            distance_squared = np.sum((xi - xj) ** 2)
            if distance_squared != 0:  # Evitar división por cero
                sum_repulsion += 1 / np.sqrt(distance_squared)
    return sum_repulsion

def h_esfera(x):
    np = len(x) // 3  # Cada punto tiene tres coordenadas
    h_x = np.zeros(np)  # Inicializa el vector de restricciones
    # Calcular la restricción para cada punto
    for k in range(np):
        h_x[k] = x[3*k]**2 + x[3*k+1]**2 + x[3*k+2]**2 - 1
    return h_x
