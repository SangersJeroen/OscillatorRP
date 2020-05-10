import matplotlib.pyplot as plt
import numpy as np

def F(t):
    return F0*t*(T-t)/(T**2)

def modulate(frequency, t):
    return np.cos(frequency*t)

t = np.linspace(0,10,1000)

T = 6
F0 = 1

F = F(t)

MASK = t > T

F[MASK] = 0

plt.plot(t, F, label=r"$F(t)$")
plt.plot(t, modulate(2*np.pi, t)*F, label="Driving Force")
plt.ylabel(r"Force $N$")
plt.xlabel(r"Time $t$ [$seconds$]")
plt.legend()
plt.show()

