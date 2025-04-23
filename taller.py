import numpy as np
import sys

e = sys.float_info.epsilon
r = 0.08
Q = 50000
A = 15000000

# Función: A*e**r*t + Q / r * (e**r*t - 1)
def equation(A, r, t, Q):
    return A * e ** (r * t) + (Q / r) * (e ** (r * t) - 1)



