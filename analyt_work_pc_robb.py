import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.integrate import odeint
from scipy.optimize import fsolve
mp.dps = 20


def np_tdgl_deriv(m1,t1):   # define the TDGL derivative with a sinusoidal applied field
    m = m1
    dmdt = -2*np_a*m - 4*np_b*m**3 + np_h0*np.cos(2*np.pi*t1/np_P)
    return dmdt
    
def mp_tdgl_deriv(t1,m1):   # define the TDGL derivative with a sinusoidal applied field
    m = m1
    dmdt = -2*mp_a*m - 4*mp_b*m**3 + mp_h0*mp.cos(2*np.pi*t1/mp_P)
    return dmdt

def np_func(m0):                          # define function which returns difference between initial
    t2 = np.linspace(0,np_P,numdivs)
    msol= odeint(np_tdgl_deriv,m0,t2)       # magnetization value and the final (integrated) magnetization
    diff = msol[-1]-m0
    avg = np.mean(msol)
    tot = abs(diff) + abs(avg)
#    print('np_func iteration: diff avg tot = ' + str(diff) + ' ' + str(avg) + ' ' +str(tot))
    return tot 

def mp_func(m0):                          # define function which returns difference between initial
    msol= mp.odefun(mp_tdgl_deriv,mp.mpf(0.0),m0)       # magnetization value and the final (integrated) magnetization
    diff = msol(mp_P) - m0
    def msol_justm(t):
        return msol(t)
    integral = mp.quad(msol_justm, [0, mp_P])
    avg = integral/mp_P
    tot = mp.fsum([diff,avg],absolute=True) 
#    print('mp_func iteration: m0 diff avg tot = ' + str(m0) + ' ' + str(diff) + ' ' + str(avg) + ' ' +str(tot))
    print(".",end="")
    return tot  

def bif_integral_search(p3):
    m0_init = -0.7
    print('bif_integral_search iteration')
    print('P = ',p3)
    global np_P
    global mp_P
    mp_P = p3
    np_P = float(p3)
    m0_soln_np = fsolve(np_func,m0_init)
    print('m0_soln_np = ',m0_soln_np)
    m0_soln_mp = mp.findroot(mp_func,m0_soln_np[0],tol=1.0e-15,solver = 'secant')
    print(".")
    print('m0_soln_mp = ',m0_soln_mp)
    msol= mp.odefun(mp_tdgl_deriv,mp.mpf(0.0),m0_soln_mp)
    # t1 = mp.linspace(mp.mpf(0.0),mp_P,100)
    # m = mp.linspace(mp.mpf(0.0),mp_P,100)
    # h = mp.linspace(mp.mpf(0.0),mp_P,100)
    # for index in range(0,100):
    #     m[index] = msol(t1[index])
    #     h[index] = mp_h0*mp.cos(2*mp.pi*t1[index]/mp_P)
    # plt.figure()
    # plt.plot(h,m)
    # plt.show()
    # plt.close()
    def cutoff_integrand(t):
        return 2*mp_a + 12*mp_b*msol(t)*msol(t)
    sqr_int = mp.quad(cutoff_integrand, [0, mp_P])
    print('calculated integral of 2a+12b*m(t)^2 = ',sqr_int)
    return sqr_int

np_P = 0.0
mp_P = mp.mpf(0.0)
P_min = mp.mpf(5.0)
P_max = mp.mpf(6.0)
mp_a = -mp.mpf(3.0)*mp.sqrt(mp.mpf(3.0))/mp.mpf(4.0)
mp_b = mp.mpf(3.0)*mp.sqrt(mp.mpf(3.0))/mp.mpf(8.0)
mp_h0 = mp.mpf(1.5)
np_a = -3.0 * np.sqrt(3.0)/4.0
np_b = 3.0 * np.sqrt(3.0)/8.0
np_h0 = 1.5
numdivs = 10000

P_c = mp.findroot(bif_integral_search,(P_min,P_max),tol=1.0e-15,solver='ridder')

p3 = P_c
m0_init = -0.7
print('P = ',p3)
mp_P = p3
np_P = float(p3)
m0_soln_np = fsolve(np_func,m0_init)
print('m0_soln_np = ',m0_soln_np)
m0_soln_mp = mp.findroot(mp_func,m0_soln_np[0],tol=1.0e-15,solver = 'secant')
print('m0_soln_mp = ',m0_soln_mp)
msol= mp.odefun(mp_tdgl_deriv,mp.mpf(0.0),m0_soln_mp)
t1 = mp.linspace(mp.mpf(0.0),mp_P,100)
m = mp.linspace(mp.mpf(0.0),mp_P,100)
h = mp.linspace(mp.mpf(0.0),mp_P,100)
for index in range(0,100):
    m[index] = msol(t1[index])
    h[index] = mp_h0*mp.cos(2*mp.pi*t1[index]/mp_P)
plt.figure()
plt.plot(h,m)
plt.show()
plt.close()
def cutoff_integrand(t):
    return 2*mp_a + 12*mp_b*msol(t)*msol(t)
sqr_int = mp.quad(cutoff_integrand, [0, mp_P])
print('calculated integral of 2a+12b*m(t)^2 = ',sqr_int)

print('Best estimate of P_c from PRL with Aaron: ',5.319357661995)
print('Estimate of P_c from this Python program = ',P_c)
