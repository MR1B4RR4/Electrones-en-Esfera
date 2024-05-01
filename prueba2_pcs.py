# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:06:45 2024

@author: Lenovo
"""

# Probar mi_pcs_local

def  f(x):
    # Función objetivo
    import numpy as np
    f = x[2]*np.cos(x[0])*np.sin(x[1])
    #f = x[0]+x[1] + x[2]
    return f
#-----------------------------------------------
    
def h(x):
    # resricciones
    import numpy as np
    h = np.zeros(2)
    h[0] = x[0]**(2) + x[1]**(2) + x[2]**(2)-1
    h[1] = x[0] + x[1] + x[2] -1
    return h
#--------------------------------------------

import numpy as np
from mi_pcs    import mi_pcs_local
from mi_pcs    import mi_pcs_local_2
x0 = np.random.rand(3)

(x,y)=mi_pcs_local(f, h, x0)
print((x,y))

print("-------------------------------")
(x,y)=mi_pcs_local_2(f, h, x0)
print((x,y))

