import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve, root_scalar

a  = -3 * mp.sqrt(3) / 4
b  =  3 * mp.sqrt(3) / 8
h1 =  mp.mpf('1.5')

num_divs = 10000

np_a = -3.0*np.sqrt(3.0)/4.0
np_b = 3.0*np.sqrt(3.0)/8.0
np_h1 = 1.5

mp_a = -mp.mpf(3)*mp.sqrt(3)/4
mp_b = mp.mpf(3)*mp.sqrt(3)/8
mp_h1 = mp.mpf('1.5')

P_c = mp.mpf('5.319357661995')
P = mp.mpf('5.3193577')

# num_
n_range = range(-32, 32 + 1)

def omega(P_):
    return 2 * mp.pi / P_

def dm_dt(t, m, P_, h0_):
    return -2*a*m - 4*b*m**3 + h1 * mp.cos(omega(P_) * t) + h0_


def get_init_m(period, h0, m0_guess=-0.55):
    np_P, mp_P = float(period), mp.mpf(period)

    def np_tdgl_deriv(m, t):
      return -2*np_a*m - 4*np_b*m**3 + np_h1*np.cos(2*np.pi*t/np_P) + h0

    def mp_tdgl_deriv(t, m):
      return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp_P) + h0

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

def get_m_k(period, h0, m0, num_k=32):
    def mp_tdgl_deriv(t, m):
      return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp.mpf(period)) + h0
    sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
    m_ks = []

    def integrand_m0(t):
      return sol(t)
    m0_mean = mp.quad(integrand_m0, [0, period], maxn=200) / period
    m_ks = {0: m0_mean}

    for k in range(num_k):
        omega_k = 2 * mp.pi * k / period

        def integrand_cos(t):
          return sol(t) * mp.cos(omega_k * t)

        def integrand_sin(t):
          return sol(t) * mp.sin(omega_k * t)

        cos_int = mp.quad(integrand_cos, [0, period], maxn=200) / period
        sin_int = mp.quad(integrand_sin, [0, period], maxn=200) / period

        m_k = (cos_int - 1j * sin_int)
        m_ks[k] = m_k

    return m_ks


m0_c = get_init_m(P, 0)
print(m0_c)
# mk_c = get_m_k(P, 0, m0_c)

h0_vals = np.logspace(-14, -8, 20)
mk_vals = []

for h0 in h0_vals:
  print(h0)
  h0_mp = mp.mpf(h0)
  m0_P = get_init_m(P, h0, m0_guess=float(m0_c))
  mk = get_m_k(P, h0_mp, m0_P, 1)
  print(mk)
  print(mp.re(mk[0]))
  mk_vals.append(mp.re(mk[0]))

# plt.plot(h0_vals, T1_vals, label=r'$|T_1|$')
# plt.plot(h0_vals, T2_vals, label=r'$T_2$')
# plt.plot(h0_vals, T3_vals, label=r'$T_3$')
# plt.legend()
# plt.ylim(-3*pow(10, -6), 3*pow(10, -6))
# plt.figure(figsize=(4.3,4))
plt.loglog(h0_vals, mk_vals, label=r'm0')
plt.loglog(h0_vals, 1e8 * h0_vals, 'k--', lw=1, label=r'$h_0$', color="black")
plt.loglog(h0_vals, pow(h0_vals, 1/3), lw=1, label=r'$h_0^{\,3}$', color="black")
# plt.xlim(pow(10, -30), pow(10, 5))
# plt.ylim(pow(10, -30), pow(10, 5))
plt.xlabel(r'$h_0$'); plt.ylabel('m0')
plt.title('Re-created Fig. 5 (Robb 2014)')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout(); plt.show()
