import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as solve

"""
|                          code in between lines                        |
"""

#Defining constants
m = 1           #Mass in kilogramme
F_naught = 1    #force in newton
freqT = 0.1     #wT dimensionless
tmax = 20       #Maximum time to elapse

#Variables
Q = 1
freq = 1
T = freqT/freq
print(T)

#Defining our timepoints
time = np.linspace(0,tmax,1000)

def force(t):
    if t < T:
        return F_naught*t*(T-t)/(T**2)
    else:
        return 0

def array_force(time):
    force_array = F_naught*time*(T-time)/(T**2)
    for i in range(0,len(time)):
        if time[i] > T:
            force_array[i] = 0
    return force_array


def system(t , func_array):
    u_prime = func_array[1]
    v_prime = 1/m * force(t) - freq/Q*func_array[1] -freq**2 *func_array[0]

    #function that takes in a vector [u, v]^T and returns [u' , v']^T

    return [u_prime, v_prime]

solution = solve(system, (0,tmax), [0,0], t_eval=time)
"""                 ^       ^       ^     \ Timepoints to evaluate at.
                    |       |        \ The initial values for u(t) and v(t)
                    |        \ The timespace in which system needs evaluating.
                     \ The system defining our system of first order ODE's
"""
"""
!!!PAS OP HIERONDER WORDT FLINK GEBEUND!!!
"""
mpl.rcParams['figure.dpi']=200

ax1 = plt.subplot(331)
plt.title(r"$Q=1$")
Q=1
wT = 0.1
T = wT/freq
solution = solve(system, (0,tmax), [0,0], t_eval=time)
plt.plot(solution.t,solution.y[0], label=r'$y(t)$')
plt.plot(time, array_force(time), label=r"$F(t)$", linestyle=":")
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylim([-0.02,0.02])
plt.ylabel(r"$\omega T = 0.1$")

ax2 = plt.subplot(332, sharex=ax1, sharey=ax1)
plt.title(r"$Q=\infty$")
Q = np.inf
wT = 0.1
T = wT/freq
solution = solve(system, (0,tmax), [0,0], t_eval=time)
plt.plot(solution.t,solution.y[0], label=r'$y(t)$')
plt.plot(time, array_force(time), label=r"$F(t)$", linestyle=":")
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.ylim([-0.02,0.02])

ax3 = plt.subplot(333, sharex=ax1)
plt.title(r"$Q=-1$")
Q = -1
wT = 0.1
T = wT/freq
solution = solve(system, (0,tmax), [0,0], t_eval=time)
plt.plot(solution.t,solution.y[0], label=r'$y(t)$')
plt.plot(time, array_force(time), label=r"$F(t)$", linestyle=":")
plt.setp(ax3.get_xticklabels(), visible=False)
#plt.ylim([-10,10])
ax3.set_yscale('symlog')
ax3.set_yticks([-1e3, 0, 1e3])

ax4 = plt.subplot(334, sharex=ax1)
Q=1
wT = 1
T = wT/freq
solution = solve(system, (0,tmax), [0,0], t_eval=time)
plt.plot(solution.t,solution.y[0], label=r'$y(t)$')
plt.plot(time, array_force(time), label=r"$F(t)$", linestyle=":")
plt.setp(ax4.get_xticklabels(), visible=False)
plt.ylim([-0.2,0.2])
plt.ylabel(r"$\omega T = 1$")
ax1.text(-10,-0.06, r"Displacement [$m$]", rotation='vertical')

ax5 = plt.subplot(335, sharex=ax1)
Q = np.inf
wT = 1
T = wT/freq
solution = solve(system, (0,tmax), [0,0], t_eval=time)
plt.plot(solution.t,solution.y[0], label=r'$y(t)$')
plt.plot(time, array_force(time), label=r"$F(t)$", linestyle=":")
plt.setp(ax5.get_xticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.ylim([-0.2,0.2])

ax6 = plt.subplot(336, sharex=ax1)
Q = -1
wT = 1
T = wT/freq
solution = solve(system, (0,tmax), [0,0], t_eval=time)
plt.plot(solution.t,solution.y[0], label=r'$y(t)$')
plt.plot(time, array_force(time), label=r"$F(t)$", linestyle=":")
plt.setp(ax6.get_xticklabels(), visible=False)
#plt.ylim([-10,10])
ax6.set_yscale('symlog')
ax6.set_yticks([-1e3, 0, 1e3])

ax7 = plt.subplot(337, sharex=ax1)
Q=1
wT = 10
T = wT/freq
solution = solve(system, (0,tmax), [0,0], t_eval=time)
plt.plot(solution.t,solution.y[0], label=r'$y(t)$')
plt.plot(time, array_force(time), label=r"$F(t)$", linestyle=":")
plt.ylim([-0.2,0.5])
plt.ylabel(r"$\omega T = 10$")

ax8 = plt.subplot(338, sharex=ax1)
Q = np.inf
wT = 10
T = wT/freq
solution = solve(system, (0,tmax), [0,0], t_eval=time)
plt.plot(solution.t,solution.y[0], label=r'$y(t)$')
plt.plot(time, array_force(time), label=r"$F(t)$", linestyle=":")
plt.setp(ax8.get_yticklabels(), visible=False)
plt.ylim([-0.2,0.5])
plt.xlabel(r"Time [$s$]")

ax9 = plt.subplot(339, sharex=ax1)
Q = -1
wT = 10
T = wT/freq
solution = solve(system, (0,tmax), [0,0], t_eval=time)
plt.plot(solution.t,solution.y[0], label=r'$y(t)$')
plt.plot(time, array_force(time), label=r"$F(t)$", linestyle=":")
#plt.ylim([-10,10])
ax9.set_yscale('symlog')
ax9.set_yticks([-1e3, 0, 1e3])

plt.gcf().set_size_inches(10,10)
#plt.tight_layout()
plt.savefig("Q3_omega_q_plot.png")
#plt.show()