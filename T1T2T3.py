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

    for k in range(-num_k, num_k+1):
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
mk_c = get_m_k(P, 0, m0_c)
print(mk_c)
print("--------")
mc = lambda k: mk_c[k]

h0_vals = 15 * np.logspace(-15, -9, 50)
T1_vals, T2_vals, T3_vals, Sig_vals = [], [], [], []

for h0 in h0_vals:
    print(h0)
    h0_mp = mp.mpf(h0)
    m0_P = get_init_m(P, h0, m0_guess=float(m0_c))
    mk = get_m_k(P, h0_mp, m0_P)
    dm = {k: mk[k]-mk_c[k] for k in n_range}
    d = lambda k: dm[k]
    # for n in range(10):
    #    print(d(n))
    #    print(d(-n))
    #    print("--------")

    S1=S2=S3=mp.mpf('0')
    for n1 in range(-16, 16 + 1):
      for n2 in range(-16, 16 + 1):
        S1 += mc(n1)*mc(n2)*d(-n1-n2)
        S2 += mc(-n1-n2)*d(n1)*d(n2)
        S3 += d(n1)*d(n2)*d(-n1-n2)
    T1 = 2*a*d(0) + 12*b*S1
    T2 = 12*b*S2
    T3 = 4*b*S3
    print(T1, T2, T3)
    print("--------")
    Sigma = T1+T2+T3

    T1_vals.append(abs(mp.re(T1)))
    T2_vals.append(abs(mp.re(T2)))
    T3_vals.append(abs(mp.re(T3)))
    Sig_vals.append(abs(mp.re(Sigma)))

print(T1_vals)
print("--------")
print(T2_vals)
print("--------")
print(T3_vals)
print("--------")

log_T1 = np.polyfit(np.log10(h0_vals), np.log10([float(t) for t in T1_vals]), 1)[0]
log_T2 = np.polyfit(np.log10(h0_vals), np.log10([float(t) for t in T2_vals]), 1)[0]
log_T3 = np.polyfit(np.log10(h0_vals), np.log10([float(t) for t in T3_vals]), 1)[0]
print(log_T1, log_T2, log_T3)

# plt.plot(h0_vals, T1_vals, label=r'$|T_1|$')
# plt.plot(h0_vals, T2_vals, label=r'$T_2$')
# plt.plot(h0_vals, T3_vals, label=r'$T_3$')
# plt.legend()
# plt.ylim(-3*pow(10, -6), 3*pow(10, -6))
# plt.figure(figsize=(4.3,4))
plt.figure(figsize=(3.5, 3.5))

# Title
plt.title(r"Scaling Terms from Eq. of Motion (Fig. 6, Robb 2014)", 
          fontsize=15, fontweight="bold")

# Axes
plt.xlabel(r"$h_0$", fontsize=22)
plt.ylabel(r"Scaling terms", fontsize=22)
plt.xscale('log')
plt.yscale('log')

# Curves
plt.plot(h0_vals, T1_vals, label=r"$|T_1|$")
plt.plot(h0_vals, T2_vals, label=r"$|T_2|$")
plt.plot(h0_vals, T3_vals, label=r"$|T_3|$")
plt.plot(h0_vals, Sig_vals, label=r"$\Sigma$")

# Reference lines
plt.plot(h0_vals, h0_vals, 'k--', lw=1, label=r"$h_0$")
plt.plot(h0_vals, 1e20 * h0_vals**3, 'k:', lw=1, label=r"$h_0^{\,3}$")

# Layout
plt.legend(loc="upper left", fontsize=13, frameon=True)
plt.tight_layout()
plt.show()
