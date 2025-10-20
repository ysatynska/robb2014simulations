# import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.optimize import root_scalar

# set number of digits for mpmath arbitrary precision
mp.dps = 20

%matplotlib inline

def np_tdgl_deriv_odeint(m1,t1):   # define the TDGL derivative with a sinusoidal applied field (numpy version)
    m = m1
    dmdt = -2*np_a*m - 4*np_b*m**3 + np_h1*np.cos(2*np.pi*t1/np_P) + np_h0
    return dmdt

def np_tdgl_deriv_solve_ivp(t1,m1):   # define the TDGL derivative with a sinusoidal applied field (numpy version)
    m = m1
    dmdt = -2*np_a*m - 4*np_b*m**3 + np_h1*np.cos(2*np.pi*t1/np_P) + np_h0
    return dmdt
    
def mp_tdgl_deriv(t1,m1):   # define the TDGL derivative with a sinusoidal applied field (mpmath version)
    m = m1
    dmdt = -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*np.pi*t1/mp_P) + mp_h0
    return dmdt

def np_func_odeint(m0):                          # define function which returns difference between initial
    t2 = np.linspace(0,np_P,numdivs)         # magnetization value and the final (integrated) magnetization
    msol= odeint(np_tdgl_deriv_odeint,m0,t2)           #  (numpy version)
    diff = msol[-1]-m0
    print('np_func_odeint iteration: m0 diff = ' + str(m0) + ' ' + str(diff))
    return diff 

def np_func_solve_ivp(m0):                          # define function which returns difference between initial
                                                    # magnetization value and the final (integrated) magnetization
    msol= solve_ivp(np_tdgl_deriv_solve_ivp,(0,np_P),m0,dense_output=True)           #  (numpy version)
    diff = msol.sol(np_P)-m0
    print('np_func_solve_ivp iteration: m0 diff = ' + str(m0) + ' ' + str(diff))
    return diff 
    
def mp_func(m0):                          # define function which returns difference between initial
    msol= mp.odefun(mp_tdgl_deriv,mp.mpf(0.0),m0)       # magnetization value and the final (integrated) magnetization
    diff = msol(mp_P) - m0                                       # (mpmath version)
    print('mp_func iteration: m0 diff  = ' + str(m0) + ' ' + str(diff))
#    print(".",end="")
    return diff
    
# set parameters of TDGL model for mpmath and numpy, as well as critical period found in another program
mp_a = -mp.mpf(3.0)*mp.sqrt(mp.mpf(3.0))/mp.mpf(4.0)
mp_b = mp.mpf(3.0)*mp.sqrt(mp.mpf(3.0))/mp.mpf(8.0)
mp_h1 = mp.mpf(1.5)
np_a = -3.0 * np.sqrt(3.0)/4.0
np_b = 3.0 * np.sqrt(3.0)/8.0
np_h1 = 1.5
P_c_np = 5.31935766199729
P_c_mp = mp.mpf(5.31935766199729)
numdivs = 10000

# create list of h0 to use in the scaling
h0_list_log10_np = np.linspace(-7.0,-4.0,7)
h0_list_len = len(h0_list_log10_np)
h0_list_10_np = np.linspace(10.,10.,h0_list_len)
h0_list_log10_mp = mp.linspace(mp.mpf(-7.0),mp.mpf(-4.0),7)
h0_list_10_mp = mp.linspace(mp.mpf(10.),mp.mpf(10.),h0_list_len)
h0_list_np = np.power(h0_list_10_np,h0_list_log10_np)
h0_list_mp = [mp.power(h0_list_10_mp[i],h0_list_log10_mp[i]) for i in range(0,h0_list_len)]
h0_target_scaling_list_mp = [mp.power(i,mp.mpf(1.0)/mp.mpf(3.0)) for i in h0_list_mp]

# specify number of mn_mag fourier components to calculate
num_fourier_comps = 8

# define functions used as integrands in calculating to Fourier components
def m0_integrand(t):
     return msol(t)
def mn_cos_integrand(t):
     return msol(t)*mp.cos(mp.mpf(2.0)*mp.pi*N*t/mp_P)
def mn_sin_integrand(t):
     return msol(t)*mp.sin(mp.mpf(2.0)*mp.pi*N*t/mp_P)

# create array to hold the fourier components computed at the critical period
mn2_mag_array_pc = np.array([mp.mpf(0) for i in range(num_fourier_comps)])

np_h0 = 0.0
mp_h0 = mp.mpf(0.0)
np_P = P_c_np
mp_P = P_c_mp
p3 = mp_P
m0_init = [-0.55]
print('P = ',p3)
#m0_soln_np = fsolve(np_func,m0_init)
m0_soln_np = root_scalar(np_func_solve_ivp, x0=m0_init, xtol=1.0e-10, method='secant')
print('m0_soln_np = ',m0_soln_np.root[0])
m0_soln_mp = mp.findroot(mp_func,[0.95*m0_soln_np.root[0],1.05*m0_soln_np.root[0]],tol=1.0e-10,solver = 'bisect')
print('m0_soln_mp = ',m0_soln_mp)
msol= mp.odefun(mp_tdgl_deriv,mp.mpf(0.0),m0_soln_mp)

m0 = (mp.mpf(1.0)/mp_P)*mp.quadsubdiv(m0_integrand, [0,mp_P])
m0_complex = m0 + 1j * 0
mn2_mag_array_pc[0] = m0*m0

for N in range(1,num_fourier_comps):
    mn_cos = (mp.mpf(2.0)/mp_P)*mp.quadsubdiv(mn_cos_integrand,[0,mp_P])
    mn_sin = (mp.mpf(2.0)/mp_P)*mp.quadsubdiv(mn_sin_integrand,[0,mp_P])
    mn_complex = 0.5*(mn_cos - 1j * mn_sin)
    mn_complex_conj = 0.5*(mn_cos + 1j * mn_sin)
#    mn_mag1 = mp.sqrt(mn_cos*mn_cos + mn_sin*mn_sin)
    mn_mag2 = mn_complex*mn_complex_conj
    mn2_mag_array_pc[N] = mp.re(mn_mag2)

print('h0 = ',mp_h0)
print('mn2_mag_array_pc = ',mn2_mag_array_pc)
print()

rows = num_fourier_comps
cols = h0_list_len
mn2_mag_array = np.array([[mp.mpf(0) for i in range(cols)] for j in range(rows)])
zn2_mag_array = np.array([[mp.mpf(0) for i in range(cols)] for j in range(rows)])
zn_mag_array = np.array([[mp.mpf(0) for i in range(cols)] for j in range(rows)])

for i in range(0,h0_list_len):
    np_h0 = h0_list_np[i]
    mp_h0 = h0_list_mp[i]
    p3 = mp_P
    m0_init = [-0.55]
    print('P = ',p3)
#    m0_soln_np = fsolve(np_func,m0_init)
    m0_soln_np = root_scalar(np_func_solve_ivp, x0=m0_init, xtol=1.0e-10, method='secant')
    print('m0_soln_np = ',m0_soln_np.root[0])
    m0_soln_mp = mp.findroot(mp_func,[0.95*m0_soln_np.root[0],1.05*m0_soln_np.root[0]],tol=1.0e-10,solver = 'bisect')
#    print('m0_soln_np = ',m0_soln_np)
#    m0_soln_mp = mp.findroot(mp_func,[0.98*m0_soln_np[0],1.02*m0_soln_np[0]],tol=1.0e-10,solver = 'bisect')
#    print('m0_soln_mp = ',m0_soln_mp)
    msol= mp.odefun(mp_tdgl_deriv,mp.mpf(0.0),m0_soln_mp)

    m0 = (mp.mpf(1.0)/mp_P)*mp.quadsubdiv(m0_integrand, [0,mp_P])
    mn2_mag_array[0,i] = m0*m0
    zn2_mag_array[0,i] = mn2_mag_array[0,i] - mn2_mag_array_pc[0]
    if (zn2_mag_array[0,i] >= 0):
        zn_mag_array[0,i] = mp.sqrt(zn2_mag_array[0,i])
    else:
        zn_mag_array[0,i] = mp.sqrt(-zn2_mag_array[0,i])
        
    for N in range(1,num_fourier_comps):
        mn_cos = (mp.mpf(2.0)/mp_P)*mp.quadsubdiv(mn_cos_integrand,[0,mp_P])
        mn_sin = (mp.mpf(2.0)/mp_P)*mp.quadsubdiv(mn_sin_integrand,[0,mp_P])
        mn_complex = 0.5*(mn_cos - 1j * mn_sin)
        mn_complex_conj = 0.5*(mn_cos + 1j * mn_sin)
#        mn_mag = mp.sqrt(mn_cos*mn_cos + mn_sin*mn_sin)
        mn_mag2 = mn_complex*mn_complex_conj
        print(str(N)+' '+str(i)+' '+str(mn_mag2))
        mn2_mag_array[N,i] = mp.re(mn_mag2)
        zn2_mag_array[N,i] = mn2_mag_array[N,i] - mn2_mag_array_pc[N]
        if (zn2_mag_array[N,i] >= 0):
            zn_mag_array[N,i] = mp.sqrt(zn2_mag_array[N,i])
        else:
            zn_mag_array[N,i] = mp.sqrt(-zn2_mag_array[N,i])
        
    
    print('i = ',i)
    print('h0 = ',mp_h0)
    print('mn2_mag_array = ',mn2_mag_array[:,i])
    print('zn2_mag_array = ',zn2_mag_array[:,i])
    print('zn_mag_array = ',zn_mag_array[:,i])
    print()

plt.figure(figsize=(4,4))
plt.title("zn vs h0 at P=Pc",fontsize=20)
plt.xlabel("h0",fontsize=18)
plt.ylabel("zn",fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.plot(h0_list_mp,h0_target_scaling_list_mp, color = 'k', label='h0 cube root')
for loop in range(0,num_fourier_comps):
    label_string = 'z'+str(loop)
    plt.plot(h0_list_mp,zn_mag_array[loop,:], label = label_string)
#plt.legend(loc=2)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.show()
plt.close()
