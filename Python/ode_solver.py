import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as solve

"""
|                          code in between lines                        |
"""

#Defining constants
m = 1
Q = -1
F_naught = 10
T = 1
freq = 1

def force(t):
    if t < T:
        return F_naught*t*(T-t)/(T**2)
    else:
        return 0


def system(t , func_array):
    u_prime = func_array[1]
    v_prime = 1/m * force(t) - freq/Q*func_array[1] -freq**2 *func_array[0]

    #function that takes in a vector [u, v]^T and returns [u' , v']^T

    return [u_prime, v_prime]

solution = solve(system, (0,30), [0,0])
plt.plot(solution.t,solution.y[0])
plt.show()