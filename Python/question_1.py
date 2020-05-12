# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:05:39 2020

@author: vanlo
"""



import numpy as np
import matplotlib.pyplot as plt

plt.clf()

w = 1.0 + 0*1j
Q = np.array([1+0*1j,1e10+0*1j, -1.0+0*1j])
F_0 = 1.0 +0*1j
m = 1.0 +0*1j

x_0 = 1.0 + 0*1j
dx_0 = 1.0 + 0*1j

D = (w/Q)**2 - 4*w**2

#test = np.array([4,-4,-9])
#print(test**(0.5))

r_1 = 1/2*(-w/Q-D**(0.5))
r_2 = 1/2*(-w/Q+D**(0.5))

c_1 = 1/(r_2 - r_1)*(-dx_0 + F_0 * Q/(w*m) + x_0*(2*r_2-r_1))
c_2 = 1/(r_2 - r_1)*(dx_0 - F_0 * Q/(w*m) -x_0*r_2)

#print(r_1)
#print(r_2)
#print(c_1)
#print(c_2)

def x(t,F_0):
    x = np.empty((len(Q),len(t))) + 1j*np.empty((len(Q),len(t)))
    
    for i in range(len(Q)):
        x[i] = c_1[i]* np.exp(r_1[i]*t) + c_2[i] * np.exp(r_2[i]*t) + F_0*Q[i]/(w**2*m)*np.sin(w*t)
    return x

t = np.linspace(0,20,200) 

x = x(t,F_0)

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.plot(t,x[0])
ax2.plot(t,x[1])
ax3.plot(t,x[2])
plt.yscale('symlog')

plt.savefig('question_1.png')
plt.show()

T = np.array([0.1, 1, 10])
F_0_array = np.empty((len(T),len(t)))
for i in range(len(T)):
    print(T[i])
    F_0_array[i,:] = (t < T[i])

x_01 = x(t,F_0_array[0,:])
x_1 = x(t,F_0_array[1,:])
x_10 = x(t,F_0_array[2,:])

fig, (ax1,ax2,ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(3,3)
ax1.plot(t,x_01[0])
ax2.plot(t,x_01[1])
ax3.plot(t,x_01[2])
plt.yscale('symlog')

ax4.plot(t,x_1[0])
ax5.plot(t,x_1[1])
ax6.plot(t,x_1[2])
plt.yscale('symlog')

ax5.plot(t,x_10[0])
ax6.plot(t,x_10[1])
ax7.plot(t,x_10[2])
plt.yscale('symlog')




    


