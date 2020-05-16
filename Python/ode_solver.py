import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as solve

"""
|                          code in between lines                        |
"""

#Defining constants
m = 1           #Mass in kilogramme
Q = 1           #Quality factor
F_naught = 1    #force in newton
freqT = 0.1     #wT dimensionless
tmax = 10       #Maximum time to elapse

#Variables
T = 1
freq = freqT/T

#Defining our timepoints
time = np.linspace(0,tmax,100)

def force(t):
    if t < T:
        return F_naught*t*(T-t)/(T**2)
    else:
        return 0


def system(t , func_array):
    u_prime = func_array[1]
    v_prime = 1/m * force(t) - freq/Q*func_array[1] -freq**2 *func_array[0]
    return [u_prime, v_prime]
    #function that takes in a vector [u, v]^T and returns [u' , v']^T

solution = solve(system, (0,tmax), [0,0], t_eval=time)
"""                 ^       ^       ^     \ Timepoints to evaluate at.
                    |       |        \ The initial values for u(t) and v(t)
                    |        \ The timespace in which system needs evaluating.
                     \ The system defining our system of first order ODE's
"""
plt.plot(solution.t,solution.y[0])
plt.show()