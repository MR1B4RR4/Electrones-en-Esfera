# Sebastian Ibarra Del Corral 193992
# Sandra Reyes Benavides 179149

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from derivadas import gradiente, mi_jacobiana, matriz_rango1
from pcs_global import mi_pcs_local_2, mi_pcs_local_3  

def f_electron(x):
    num_points = 21  # Total de puntos incluyendo el anclado
    x_full = np.zeros(3 * num_points)
    x_full[0:3] = np.array([1, 0, 0])  # Punto anclado
    x_full[3:] = x  # Los puntos restantes
    sum_repulsion = 0
    for i in range(num_points):
        for j in range(i + 1, num_points):
            xi = x_full[3*i:3*i+3]
            xj = x_full[3*j:3*j+3]
            distance_squared = np.sum((xi - xj) ** 2)
            if distance_squared != 0:
                sum_repulsion += 1 / np.sqrt(distance_squared)
    return sum_repulsion

def h_esfera(x):
    num_points_to_optimize = 20  # Puntos a optimizar, sin incluir el anclado
    h_x = np.zeros(num_points_to_optimize)
    for k in range(num_points_to_optimize):
        h_x[k] = x[3*k]**2 + x[3*k+1]**2 + x[3*k+2]**2 - 1
    return h_x

# Generar puntos iniciales aleatorios en la esfera para 20 puntos
num_points_random = 20
initial_points = np.random.randn(3 * num_points_random)
initial_norms = np.linalg.norm(initial_points.reshape(num_points_random, 3), axis=1)
initial_points /= initial_norms[:, np.newaxis].repeat(3, axis=1).flatten()

# Resolver el problema de optimización
optimized_points, y_opt = mi_pcs_local_3(f_electron, h_esfera, initial_points)

if optimized_points is not None:
    optimized_points = optimized_points.reshape((num_points_random, 3))
    print("Optimized Points:\n", optimized_points)
else:
    print("Optimization failed")

# Suponiendo que 'optimized_points' es el array obtenido del proceso de optimización,
# y añadimos el punto anclado:
optimized_points = np.vstack(([1, 0, 0], optimized_points))  # Asegúrate de que 'optimized_points' está definido

# Crear una figura para la graficación
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dibujar la esfera
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, color='b', alpha=0.3)

# Dibujar los puntos
ax.scatter(optimized_points[:, 0], optimized_points[:, 1], optimized_points[:, 2], color='r', s=50)

# Etiquetas y título
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('Points on Sphere')

# Mostrar la gráfica
plt.show()
