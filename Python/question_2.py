# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:27:33 2020

@author: vanlo
"""



import numpy as np
import matplotlib.pyplot as plt

plt.clf()

w = 1.0 + 0*1j
Q = np.array([1+0*1j,1e10+0*1j, -1.0+0*1j])
T = np.array([0.1, 1, 10])
F_0 = 1.0 +0*1j
m = 1.0 +0*1j

x_0 = 1
dx_0 = 1

t_1 = 0
t_2_q1 = [10,20,40]
t_2_q2 = [20,20,20,20,20,20,20,20,20]
N = 200



Q = np.array([1+0*1j,1e10+0*1j, -1.0+0*1j])


def f(t,Q,F_0):
    D = (w/Q)**2 - 4*w**2
    
    r_1 = 1/2*(-w/Q-D**(0.5))
    r_2 = 1/2*(-w/Q+D**(0.5))

    c_1 = 1/(r_2 - r_1)*(-dx_0 + F_0 * Q/(w*m) + x_0*(2*r_2-r_1))
    c_2 = 1/(r_2 - r_1)*(dx_0 - F_0 * Q/(w*m) -x_0*r_2)
    
    x = c_1* np.exp(r_1*t) + c_2 * np.exp(r_2*t) + F_0*Q/(w**2*m)*np.sin(w*t)
    
    dx = c_1*r_1*np.exp(r_1*t) + c_2 *r_2* np.exp(r_2*t) + F_0*Q/(w*m)*np.cos(w*t)
    return [x,dx]




x_1 = np.empty((len(Q),N))
for i in range(len(Q)):
    t = np.linspace(t_1,t_2_q1[i],N) 
    x_1[i,:] = f(t,Q[i],F_0)[0]

x_2 = np.empty((len(T)*len(Q),N))
for i in range(len(T)):
    for i in range(len(Q)):
        print(i)
        
        x_0_i = f(T[i],Q[i],F_0)[0]
        dx_0_i = f(T[i],Q[i],F_0)[1]
        
        t = np.linspace(0,t_2_q2,N) 
        t_critical = np.sum(t<T[i])+1
        print(t_critical)
        
        x_2[i,:t_critical] = f(t[:t_critical],Q[i],F_0)[0]
        x_2[i,t_critical:] = f(t[t_critical:]-T,Q[i],0)[0]
    


    

#T = np.array([0.1, 1, 10])
#F_0_array = np.empty((len(T),len(t)))
#for i in range(len(T)):
#    print(T[i])
#    F_0_array[i,:] = (t < T[i])
#
#x_01 = x(t,F_0_array[0,:])
#x_1 = x(t,F_0_array[1,:])
#x_10 = x(t,F_0_array[2,:])
#
#fig, (ax1,ax2,ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(3,3)
#ax1.plot(t,x_01[0])
#ax2.plot(t,x_01[1])
#ax3.plot(t,x_01[2])
#plt.yscale('symlog')
#
#ax4.plot(t,x_1[0])
#ax5.plot(t,x_1[1])
#ax6.plot(t,x_1[2])
#plt.yscale('symlog')
#
#ax5.plot(t,x_10[0])
#ax6.plot(t,x_10[1])
#ax7.plot(t,x_10[2])
#plt.yscale('symlog')




    


