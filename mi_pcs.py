# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:41:15 2024

@author: Zeferino Parada García
         Optimización Numérica
         ITAM
"""

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
    import numpy as np
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
          
            
        
    
    
    
    
    
