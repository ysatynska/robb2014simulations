import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.integrate import odeint
from scipy.optimize import fsolve, root_scalar

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

h_mults = np.logspace(-11, -1, num=10)
z_k_all = [[mp.mpf(0) for _ in h_mults] for _ in range(num_k)]

# h0 = .3
# h2_c = .4
# h2_s = .6
# h4_c = .2
# h4_s = .7

h0 = 0.458511002351287
h2_c = 0.610162246721079
h2_s = 0.25005718740867244
h4_c = 0.4011662863159199
h4_s = 0.3233779454085565

def get_init_m(period, h_mult=0.0):
    np_P, mp_P = float(period), mp.mpf(period)
    m0_guess=-0.55

    def np_tdgl_deriv(m, t):
      return -2*np_a*m - 4*np_b*m**3 + np_h1*np.cos(2*np.pi*t/np_P) + np_h1*np.cos(3*2*np.pi*t/np_P) + np_h1*np.sin(2*np.pi*t/np_P) + np_h1*np.sin(3 * 2*np.pi*t/np_P) + h_mult * (h0 + h2_c*np.cos(2*2*np.pi*t/np_P) + h2_s*np.sin(2*2*np.pi*t/np_P) + h4_c*np.cos(4*2*np.pi*t/np_P) + h4_s*np.sin(4*2*np.pi*t/np_P))

    def mp_tdgl_deriv(t, m):
      return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp_P) + mp_h1*mp.cos(3*2*mp.pi*t/mp_P) + mp_h1*mp.sin(2*mp.pi*t/mp_P) + mp_h1*mp.sin(3 * 2*mp.pi*t/mp_P) + h_mult * (h0 + h2_c*mp.cos(2*2*mp.pi*t/mp_P) + h2_s*mp.sin(2*2*mp.pi*t/mp_P) + h4_c*mp.cos(4*2*mp.pi*t/mp_P) + h4_s*mp.sin(4*2*mp.pi*t/mp_P))

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
    print(sol)
    print(f(sol.root))
    if not sol.converged:
      raise RuntimeError("root_scalar did not converge")

    return mp.mpf(sol.root)

def get_m_k(m0, period, h_mult = 0):
    def mp_tdgl_deriv(t, m):
      return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp.mpf(period)) + mp_h1*mp.cos(3*2*mp.pi*t/mp.mpf(period)) + mp_h1*mp.sin(2*mp.pi*t/mp.mpf(period)) + mp_h1*mp.sin(3 * 2*mp.pi*t/mp.mpf(period)) + h_mult * (h0 + h2_c*mp.cos(2*2*mp.pi*t/mp.mpf(period)) + h2_s*mp.sin(2*2*mp.pi*t/mp.mpf(period)) + h4_c*mp.cos(4*2*mp.pi*t/mp.mpf(period)) + h4_s*mp.sin(4*2*mp.pi*t/mp.mpf(period)))

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
  print(j, init_m_c)

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

plt.plot(h_mults, [mp.power(i,mp.mpf(1.0)/mp.mpf(3.0)) for i in h_mults], 'k--', lw=1, label=r"$h_{mult}^{1/3}$")
for k in range(num_k):
    plt.plot(h_mults, [float(z) for z in z_k_all[k]], label=fr"$z_{{{k}}}$")
plt.tight_layout()
plt.legend(loc="upper left")
plt.show()
