# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:27:33 2020

@author: vanlo
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter as scalarformatter

plt.close('all')

w = 2*np.pi*(1.0 + 0*1j)
Q = np.array([1+0*1j,1e10+0*1j, -1.0+0*1j])
T = np.array([0.1, 1, 10])/w
F_0 = 1.0 +0*1j
m = 1.0 +0*1j

t_1 = 0
t_2_q1 = [10,10,10]
t_2_q2 = np.array([[3,3,3],[5,5,5],[5,5,5]])
N = int(1e3)



Q = np.array([1+0*1j,1e10+0*1j, -1.0+0*1j])

def force(t):
    F = F_0 * np.cos(w*t)
    return F


def f(t,Q,F_0,x_0,dx_0):
    D = (w/Q)**2 - 4*w**2
    
    r_1 = 1/2*(-w/Q-D**(0.5))
    r_2 = 1/2*(-w/Q+D**(0.5))

    c_1 = 1/(r_2 - r_1)*(-dx_0 + F_0 * Q/(w*m) + x_0*(2*r_2-r_1))
    c_2 = 1/(r_2 - r_1)*(dx_0 - F_0 * Q/(w*m) -x_0*r_2)
    
    x = c_1* np.exp(r_1*t) + c_2 * np.exp(r_2*t) + F_0*Q/(w**2*m)*np.sin(w*t)
    dx = c_1*r_1*np.exp(r_1*t) + c_2 *r_2* np.exp(r_2*t) + F_0*Q/(w*m)*np.cos(w*t)
    return [x,dx]

t_q1 = np.transpose( np.linspace(t_1,t_2_q1,N) )
x_0 = 0
dx_0 = 0

x_1 = np.empty((len(Q),N))
force_scale_1 = np.array([1e-2,4e-1,1e5])

for i in range(len(Q)):
    x_1[i,:] = f(t_q1[i,:],Q[i],F_0,x_0,dx_0)[0]


x_0 = 0
dx_0 = 0

count = 0
t_q2 = np.transpose(np.linspace(t_1,t_2_q2,N))
t_critical = np.empty((len(Q),len(T)))



x_2 = np.empty((len(T),len(Q),N))

force_plot = np.copy(x_2)
force_scale = np.array([[1e-3,1e-3,1e1],[1e-2,1e-2,1e1],[1e-2,1e-1,1e2]])

for i in range(len(T)):
    for j in range(len(Q)):
        
        x_0_i = f(T[i],Q[j],F_0,x_0,dx_0)[0]
        dx_0_i = f(T[i],Q[j],F_0,x_0,dx_0)[1]
        
        t_critical[i,j] = next(k for k,n in enumerate(t_q2[i,j,:]) if n >T[i])
        t_i = int(t_critical[i,j])

        x_2_driven = f(t_q2[i,j,:t_i],Q[j],F_0,x_0,dx_0)[0]
        x_2_free = f((t_q2[i,j,t_i:]-T[i]),Q[j],0,x_0_i,dx_0_i)[0]
        
        x_2[i,j,:t_i] = x_2_driven
        x_2[i,j,t_i:] = x_2_free
        
        force_plot[i,j,:t_i] = force(t_q2[i,j,:t_i])*force_scale[i,j]
        force_plot[i,j,t_i:] = t_q2[i,j,t_i:] * 0
        
    


fig, ax = plt.subplots(3,1, sharex=True)
fig.set_size_inches(12, 6)

ax[-1].set_xlabel(' ', color=(0, 0, 0, 0))
ax[-1].set_ylabel('\n ', color=(0, 0, 0, 0))
#
fig.text(0.55, 0.04, '$t  \; \; [rad/2 \pi] \; \; \; \u2192$', va='center', ha='center')
fig.text(0.02, 0.5, '$x \; [m] \; \; \; \u2192 $', va='center', ha='center', rotation='vertical')

for i in range(len(Q)):
    ax[i].plot(t_q1[i,:],x_1[i,:])
    ax[i].plot(t_q1[i,:],force(t_q1[i,:])*force_scale_1[i], linestyle=':')
    ax[i].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    ax[i].yaxis.major.formatter._useMathText = True
    ax[i].margins(x=0)
    
    

ax[0].set_title('$Q = 1$')

#ax[1].set_yticks([-0.6-0.2,0,0.2])
ax[1].set_title('$Q \; \u2192 \; \infty $')

ax[2].set_title('$Q = -1$')
ax[2].set_yscale('symlog')
ax[2].set_yticks([-1e8,-1e4,0,1e4,1e8])


fig.tight_layout()

plt.savefig(r'C:\Users\vanlo\Documents\GitHub\OscillatorRP\Verslag\figures\graph_q1.png', dpi=100)
plt.show()




fig, ax = plt.subplots(3,3,gridspec_kw={'width_ratios': [1.5,2,2]}, sharex='col')
fig.set_size_inches(14, 8)


fig.text(0.55, 0.04, '$t  \; \; [rad/2 \pi] \; \; \; \u2192$', va='center', ha='center')
fig.text(0.2, 0.93, '$Q = 1$', va='center', ha='center')
fig.text(0.5, 0.93, '$Q \; \u2192 \; \infty $', va='center', ha='center')
fig.text(0.83, 0.93, '$Q = -1$', va='center', ha='center')

fig.text(0.09, 0.45, '$x \; [m] \; \; \; \u2192 $', va='center', ha='center', rotation='vertical')
fig.text(0.06, 0.78, '$\omega T = 0.1$', va='center', ha='center')
fig.text(0.05, 0.5, '$\omega T = 1$', va='center', ha='center')
fig.text(0.06, 0.22, '$\omega T = 10$', va='center', ha='center')


for i in range(len(T)):
    for j in range(len(Q)):
        ax[i,j].plot(t_q2[i,j,:],x_2[i,j,:])
        ax[i,j].plot(t_q2[i,j,:],force_plot[i,j,:], linestyle=':')
        ax[i,j].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
        ax[i,j].yaxis.major.formatter._useMathText = True
        ax[i,j].margins(x=0)
        
        ax[i,0].set_ylabel('\n \n \n \n  \n \n ', color=(0, 0, 0, 0))
        ax[2,i].set_xlabel(' \n  \n  ', color=(0, 0, 0, 0))
        ax[0,i].set_title(' \n \n \n \n  ', color=(0, 0, 0, 0))


ax[0,0].set_yticks([0, 0.0005, 0.001])
ax[0,2].set_yscale('symlog')
ax[0,2].set_yticks([-1e3, -1e1, 0, 1e1, 1e3])


ax[1,0].set_yticks([0,0.004, 0.008])
ax[1,2].set_yscale('symlog')
ax[1,2].set_yticks([-1e3, -1e1, 0, 1e1, 1e3])

ax[2,2].set_yscale('symlog')
ax[2,2].set_yticks([-1e4, -1e2, 0, 1e2, 1e4])

plt.tight_layout()
plt.savefig(r'C:\Users\vanlo\Documents\GitHub\OscillatorRP\Verslag\figures\graph_q2.png', dpi=100)
plt.show()




    


