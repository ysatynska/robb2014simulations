import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.integrate import odeint
from scipy.optimize import fsolve

mp.dps = 20
num_divs = 10000
num_k = 8

np_a = -3.0*np.sqrt(3.0)/4.0
np_b = 3.0*np.sqrt(3.0)/8.0
np_h1 = 1.5

mp_a = -mp.mpf(3)*mp.sqrt(3)/4
mp_b = mp.mpf(3)*mp.sqrt(3)/8
mp_h1 = mp.mpf('1.5')

P_c = 2.31509101536556
P_c_mp = mp.mpf(P_c)

def get_init_m(period, m0_guess = -.55):
  np_P, mp_P = period, mp.mpf(period)
  def np_tdgl_deriv(m, t):
    return -2*np_a*m - 4*np_b*m**3 + np_h1*np.cos(2*np.pi*t/np_P) + np_h1*np.cos(3*2*np.pi*t/np_P) + np_h1*np.sin(2*np.pi*t/np_P) + np_h1*np.sin(3 * 2*np.pi*t/np_P)

  def mp_tdgl_deriv(t, m):
    return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp_P) + mp_h1*mp.cos(3*2*mp.pi*t/mp_P) + mp_h1*mp.sin(2*mp.pi*t/mp_P) + mp_h1*mp.sin(3 * 2*mp.pi*t/mp_P)

  def np_func(m0):
    msol= odeint(np_tdgl_deriv, m0, np.linspace(0,np_P,num_divs))
    return msol[-1]-m0
  m0_np = fsolve(np_func, m0_guess, xtol=1e-8)[0]

  def residue(m0):
    sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0.0), m0)
    return sol(mp_P) - m0
  return mp.findroot(residue, [0.98*m0_np, 1.02*m0_np], solver='bisect', tol=1e-10)

def get_m_k(m0, period):
    def mp_tdgl_deriv(t, m):
      return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp.mpf(period))
    sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
    m_k_cs_2 = []

    def integrand_m0(t):
      return sol(t)
    m0_mean = mp.quad(integrand_m0, [0, period], maxn=200) / period
    m_k_cs_2.append(m0_mean**2)

    for k in range(1, num_k):
        omega_k = 2 * mp.pi * k / period

        def integrand_cos(t):
          return sol(t) * mp.cos(omega_k * t)

        def integrand_sin(t):
          return sol(t) * mp.sin(omega_k * t)

        cos_int = mp.quad(integrand_cos, [0, period], maxn=200) / period
        sin_int = mp.quad(integrand_sin, [0, period], maxn=200) / period

        m_k = (cos_int - 1j * sin_int)
        m_k_cs_2.append(mp.re(m_k * mp.conj(m_k)))

    return m_k_cs_2

init_m_c = get_init_m(P_c)
print("init_m_c:", init_m_c)
m_k_c_2 = get_m_k(init_m_c, P_c_mp)
print(m_k_c_2)

epsilons = np.logspace(-9, -7, 30)
P_vals = P_c * (1 - epsilons)

eps_mp = [mp.mpf(e) for e in epsilons]
z_k_all = [[mp.mpf(0) for _ in epsilons] for _ in range(num_k)]

m0_prev = init_m_c
for j, P in enumerate(P_vals):
  m0 = get_init_m(P, float(m0_prev))
  m0_prev = m0
  m_k_2 = get_m_k(m0, mp.mpf(P))

  for k in range(num_k):
    diff = abs(m_k_2[k] - m_k_c_2[k])
    z_k_all[k][j] = diff

plt.figure(figsize=(3.5, 3.5))
plt.title(r"$z_k$ va $\varepsilon$")
plt.xlabel(r"$\varepsilon$")
plt.ylabel(r"$z_k$")
plt.xscale('log'); plt.yscale('log')

plt.plot(epsilons, epsilons, 'k--', lw=1, label=r"$\varepsilon$")
plt.plot(epsilons, [mp.power(i, 1/2) for i in epsilons], 'k--', lw=1, label=r"$\varepsilon^{1/2}$")
for k in range(num_k):
    plt.plot(epsilons, [float(z) for z in z_k_all[k]], label=fr"$z_{{{k}}}$")
plt.legend(fontsize=7)
plt.tight_layout()
plt.show()
