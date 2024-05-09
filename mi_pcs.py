# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:41:15 2024

@author: Zeferino Parada García
         Optimización Numérica
         ITAM
"""
import numpy as np
from derivadas import gradiente
from derivadas import mi_jacobiana
from derivadas import matriz_rango1

def mi_pcs_local(fun, h, x):
    # Programación Cuadrática Sucesiva Local con actualización de
    # Powell para el problema
    #   Min     fun(x)
    # Sujeto a  h(x) = 0
    #
    # donde fun:R^n --> R   y h:R^n --R^m 
    #  son dos veces continuamente diferenciables.
    # el vector x de entrada es el punto inical.
    # El método que se codifica es el método de Newton a las 
    # condiciones necesarias de primer orden para un mínimo local
    # del problema de minimización.
    #---------------------------------------------
    # Optimización Numérica
    #   ITAM
    # 17 de abril de 2024
    #----------------------------------------------
        
    tol = 10**(-5)
    maxiter = 10
    iter = 0
    #--------------------------
    h_x = h(x)
    n = len(x)
    m = len(h_x)
    grad_x = gradiente(fun,x)
    J_x = mi_jacobiana(h,x)
    y = np.zeros(m)
    B = np.identity(n)
    M = np.zeros((n+m,n+m))
    #---condiciones necesarias de primer orden--------
    cnpo1 = grad_x + np.dot(J_x.T, y)
    cnpo = np.concatenate((cnpo1, h_x), 0)
    cnpo_norma = np.linalg.norm(cnpo)
    #-----------parte iterativa-----------------------
    while(cnpo_norma > tol and iter<maxiter):
        iter = iter + 1
        #  Matriz del sistema Lineal
        M[:n,0:n] = B
        M[0:n, n:n+m] = J_x.T
        M[n:n+m,0:n] = J_x
        # Lado derecho del sistema lineal
        ld = np.concatenate((grad_x, h_x),0)
        # Solución del sistema lineal
        w = np.linalg.solve(M, -ld)
        #------------------------------
        #  Actualización de 
        d_x = w[0:n]
        d_y = w[n:n+m]
        x_trial = x + d_x
        y_trial = d_y
        #-------------------------------------------
        grad_x_trial = gradiente(fun,x_trial)
        J_x_trial = mi_jacobiana(h,x_trial)
        s = x_trial-x
        v = grad_x_trial + np.dot(J_x_trial.T, y)-cnpo1
        #----------------------------------------------
        #   Actualización de Powell
        ang = np.dot(s.T,v)
        s_aux = np.dot(B,s)
        B_ang = np.dot(s, s_aux)  # (s.T)*B*s
        theta = 1.0
        if(ang<(0.2)*B_ang):
             theta = (0.8)*B_ang/(B_ang-ang)
             
        rv = theta*v - (1-theta*(s_aux))
         # ------------------------------------------------------
        den1 = np.dot(s,rv)
        B_1 = matriz_rango1(rv,rv)
        B_2 = matriz_rango1(s_aux,s_aux)
        B = B +(1/den1)*(B_1)-(1/B_ang)*(B_2)
       #--------------------------------------------------- 
        
        
        
        #--------------------------------------------------
        x =x_trial
        y = y_trial
        grad_x = grad_x_trial
        J_x = J_x_trial
        h_x = h(x)
        cnpo1 = grad_x + np.dot(J_x.T,y)
        cnpo = np.concatenate((cnpo1,h_x),0)
        cnpo_norma = np.linalg.norm(cnpo)
        print(iter,"|",cnpo_norma)
        if(cnpo_norma <= tol or iter==maxiter):
            return x,y
        
        #-----------------------------------------------------------
                
def mi_pcs_local_2(fun, h, x):
      # Programación Cuadrática Sucesiva Local con actualización de
      # Powell para el problema
      #   Min     fun(x)
      # Sujeto a  h(x) = 0
      #
      # donde fun:R^n --> R   y h:R^n --R^m 
      #  son dos veces continuamente diferenciables.
      # el vector x de entrada es el punto inical.
      # El método que se codifica es el método de Newton a las 
      # condiciones necesarias de primer orden para un mínimo local
      # del problema de minimización.
      #---------------------------------------------
      # Optimización Numérica
      #   ITAM
      # 17 de abril de 2024
      #----------------------------------------------
      import numpy as np
      import scipy
      from derivadas import gradiente
      from derivadas import mi_jacobiana
      from derivadas import matriz_rango1
      
      tol = 10**(-5)
      maxiter = 10
      iter = 0
      #--------------------------
      h_x = h(x)
      n = len(x)
      m = len(h_x)
      grad_x = gradiente(fun,x)
      J_x = mi_jacobiana(h,x)
      y = np.zeros(m)
      B = np.identity(n)
      M = np.zeros((n+m,n+m))
      #---condiciones necesarias de primer orden--------
      cnpo1 = grad_x + np.dot(J_x.T, y)
      cnpo = np.concatenate((cnpo1, h_x), 0)
      cnpo_norma = np.linalg.norm(cnpo)
      #-----------parte iterativa-----------------------
      while(cnpo_norma > tol and iter<maxiter):
          iter = iter + 1
          #  Matriz del sistema Lineal
          M[:n,0:n] = B
          M[0:n, n:n+m] = J_x.T
          M[n:n+m,0:n] = J_x
          # Lado derecho del sistema lineal
          ld = np.concatenate((grad_x, h_x),0)
          # Solución del sistema lineal
          w = np.linalg.solve(M, -ld)
          #------------------------------
          #  Actualización de 
          d_x = w[0:n]
          d_y = w[n:n+m]
          x_trial = x + d_x
          y_trial = d_y
          #-------------------------------------------
          grad_x_trial = gradiente(fun,x_trial)
          J_x_trial = mi_jacobiana(h,x_trial)
          s = x_trial-x
          v = grad_x_trial + np.dot(J_x_trial.T, y)-cnpo1
          #----------------------------------------------
          #   Actualización de Powell
          ang = np.dot(s.T,v)
          s_aux = np.dot(B,s)
          B_ang = np.dot(s, s_aux)  # (s.T)*B*s
          theta = 1.0
          if(ang<(0.2)*B_ang):
               theta = (0.8)*B_ang/(B_ang-ang)
               
          rv = theta*v - (1-theta*(s_aux))
           # ------------------------------------------------------
          den1 = np.dot(s,rv)
          B_1 = matriz_rango1(rv,rv)
          B_2 = matriz_rango1(s_aux,s_aux)
          B = B +(1/den1)*(B_1)-(1/B_ang)*(B_2)
         #--------------------------------------------------- 
         # Actualizaciones
          x =x_trial
          y = y_trial
          grad_x = grad_x_trial
          J_x = J_x_trial
          h_x = h(x)
          #-----------Lagrangeano con mínimos cuadrados lineales-----------------------------
          rango_J = np.linalg.matrix_rank(J_x)
          if(rango_J == m):
              J_matrix = np.dot(J_x, J_x.T)
              r_hand = np.dot(J_x, grad_x)
              y = np.linalg.solve(J_matrix,r_hand)
          #--------------------------------------------------
          cnpo1 = grad_x + np.dot(J_x.T,y)
          cnpo = np.concatenate((cnpo1,h_x),0)
          cnpo_norma = np.linalg.norm(cnpo)
          print(iter,"|",cnpo_norma)
          if(cnpo_norma <= tol or iter==maxiter):
              return x,y
          
          #-----------------------------------------------------------
          

def line_search_wolfe(f, x, p, g, c1=1e-4, c2=0.9, max_line_search=100):
    alpha = 1e-4
    alpha_min, alpha_max = 0, np.inf

    for _ in range(max_line_search):
        x_new = x + alpha * p
        f_new = f(x_new)
        # Condición de Armijo
        if f_new > f(x) + c1 * alpha * np.dot(g, p):
            alpha_max = alpha
            alpha = (alpha_min + alpha_max) / 2
            continue
        # Condición de curvatura de Wolfe
        g_new = gradiente(f, x_new)
        if np.dot(g_new, p) < c2 * np.dot(g, p):
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha = 2 * alpha_min
            else:
                alpha = (alpha_min + alpha_max) / 2
            continue
        return alpha, True
    return None, False  # Retorna None si no se encuentra un alpha adecuado


def mi_pcs_local_3(fun, h, x):
    tol = 1e-6
    maxiter = 100
    iter = 0
    n = len(x)
    h_x = h(x)
    m = len(h_x)
    grad_x = gradiente(fun, x)
    J_x = mi_jacobiana(h, x)
    y = np.zeros(m)
    B = np.identity(n)
    M = np.zeros((n+m, n+m))
    
    cnpo1 = grad_x + np.dot(J_x.T, y)
    cnpo = np.concatenate((cnpo1, h_x), 0)
    cnpo_norma = np.linalg.norm(cnpo)

    while cnpo_norma > tol and iter < maxiter:
        iter += 1
        # Matriz del sistema lineal
        M[:n, :n] = B
        M[:n, n:n+m] = J_x.T
        M[n:n+m, :n] = J_x

        # Lado derecho del sistema lineal
        ld = np.concatenate((grad_x, h_x), 0)

        # Solución del sistema lineal
        w = np.linalg.solve(M, -ld)
        d_x = w[:n]
        d_y = w[n:n+m]

        # Búsqueda de línea con condiciones de Wolfe
        alpha, ok = line_search_wolfe(fun, x, d_x, grad_x, c1=1e-4, c2=0.9)
        if not ok:
            print("Búsqueda de línea fallida. Ajustando alfa manualmente. alpha anterior: ")
            alpha = 10**-4 # Ajuste manual si la búsqueda de línea falla

        # Actualizaciones de x y y
        x += alpha * d_x
        y += alpha * d_y

        # Recalcular gradiente y jacobiana
        grad_x = gradiente(fun, x)
        J_x = mi_jacobiana(h, x)
        h_x = h(x)

        # Actualización de la matriz B usando Powell
        s = alpha * d_x
        v = grad_x + np.dot(J_x.T, y) - cnpo1
        # Actualización de Powell
        rv = v - np.dot(B, s)
        if np.dot(s, rv) > 0:
            B += np.outer(rv, rv) / np.dot(s, rv)

        # Reevaluar condiciones
        cnpo1 = grad_x + np.dot(J_x.T, y)
        cnpo = np.concatenate((cnpo1, h_x), 0)
        cnpo_norma = np.linalg.norm(cnpo)
        
        print(f"Iteración {iter}: Norma CNPO = {cnpo_norma} Alfa: {alpha}")

        if cnpo_norma <= tol:
            break

    return x, y