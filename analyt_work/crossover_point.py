import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.integrate import odeint
from scipy.optimize import fsolve, root_scalar

mp.dps = 20
num_divs = 10000
num_k = 8

np_a = -3.0/2.0
np_b = 1.0/2.0
np_h1 = 3.0

mp_a = -mp.mpf(3)/2
mp_b = mp.mpf(1)/2
mp_h1 = mp.mpf('3.0')

P_c = 5.31935766199729
P_c_mp = mp.mpf(P_c)

h_mults = np.logspace(-1, -10, num=10)
z_k_all = [[mp.mpf(0) for _ in h_mults] for _ in range(num_k)]

h0 = .3
h2 = .4
h4 = .2

def get_init_m(period, h_mult=0.0):
    np_P, mp_P = float(period), mp.mpf(period)
    m0_guess=-0.55

    def np_tdgl_deriv(m, t):
      return -2*np_a*m - 4*np_b*m**3 + np_h1*np.cos(2*np.pi*t/np_P) + h_mult * (h0 + h2*np.cos(2*2*np.pi*t/np_P) + h4*np.cos(4*2*np.pi*t/np_P))

    def mp_tdgl_deriv(t, m):
      return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp_P) + h_mult * (h0 + h2*mp.cos(2*2*mp.pi*t/mp_P) + h4*mp.cos(4*2*mp.pi*t/mp_P))

    def np_func(m0):
      msol = odeint(np_tdgl_deriv, m0, np.linspace(0, np_P, num_divs))
      return msol[-1] - m0

    m0_seed = fsolve(np_func, m0_guess, xtol=1e-6)[0]

    def residue(mp_m0):
      sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), mp_m0)
      return sol(mp_P) - mp_m0

    def f(m):
      return float(residue(m))

    step = 1e-4
    a, b = m0_seed - step, m0_seed + step
    fa, fb = f(a), f(b)

    for _ in range(25):
      if np.sign(fa) == 0:
        return mp.mpf(a)
      if np.sign(fb) == 0:
        return mp.mpf(b)
      if np.sign(fa) != np.sign(fb):
        break
      step *= 1.6
      a, b = m0_seed - step, m0_seed + step
      fa, fb = f(a), f(b)

    sol = root_scalar(f, bracket=[a, b], method='brentq', xtol=1e-10)
    if not sol.converged:
      raise RuntimeError("root_scalar did not converge")

    return mp.mpf(sol.root)

def get_m_k(m0, period, h_mult = 0):
    def mp_tdgl_deriv(t, m):
      return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp.mpf(period)) + h_mult * (h0 + h2*mp.cos(2*2*mp.pi*t/period) + h4*mp.cos(4*2*mp.pi*t/period))
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
m_k_c_2 = get_m_k(init_m_c, P_c_mp)

for j, h_mult in enumerate(h_mults):
  init_m_c = get_init_m(P_c, h_mult)

  m_k_2 = get_m_k(init_m_c, mp.mpf(P_c), h_mult)

  for k in range(num_k):
    diff = abs(m_k_2[k] - m_k_c_2[k])
    print(diff)
    z_k_all[k][j] = mp.sqrt(diff)

plt.figure(figsize=(3.5, 3.5))
plt.title(r"$z_k$ va $h_{mult}$")
plt.xlabel(r"$h_{mult}$")
plt.ylabel(r"$z_k$")
plt.xscale('log'); plt.yscale('log')

plt.plot(h_mults, [mp.power(i,mp.mpf(1.0)/mp.mpf(3.0)) for i in h_mults], lw=1, label=r"$h_{mult}^{1/3}$")
# plt.plot(h_mults, [mp.power(i,mp.mpf(1.0)/mp.mpf(2.0)) for i in h_mults], lw=1, label=r"$h_{mult}^{1/2}$")
# plt.plot(h_mults, [mp.power(i,mp.mpf(1.0)/mp.mpf(1.0)) for i in h_mults], lw=1, label=r"$h_{mult}^{1/1}$")
for k in range(num_k):
    plt.plot(h_mults, [float(z) for z in z_k_all[k]], label=fr"$z_{{{k}}}$")
plt.tight_layout()
plt.legend(loc="upper left")
plt.show()
