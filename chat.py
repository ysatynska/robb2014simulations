import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

mp.dps = 20
num_divs = 10000
num_k = 8

np_a = -3.0 * np.sqrt(3.0)/4.0
np_b = 3.0 * np.sqrt(3.0)/8.0
np_h0 = 1.5

mp_a = -mp.mpf(3) * mp.sqrt(3) / 4
mp_b = mp.mpf(3) * mp.sqrt(3) / 8
mp_h0 = mp.mpf('1.5')

P_c = 5.31935766199729
mp_P_c = mp.mpf(P_c)

def np_tdgl_deriv(m1, t1):
  return -2*np_a*m1 - 4*np_b*m1**3 + np_h0*np.cos(2*np.pi*t1/np_P)

def mp_tdgl_deriv(t1, m1):
  return -2*mp_a*m1 - 4*mp_b*m1**3 + mp_h0*mp.cos(2*mp.pi*t1/mp_P)

def np_func(m0):
  t = np.linspace(0.0, np_P, num_divs)
  m = odeint(np_tdgl_deriv, m0, t)
  return m[-1, 0] - m0

def mp_func(m0):
  sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
  return sol(mp_P) - m0

def solve_m0(P: float, m0_guess: float = -0.55):
    global np_P, mp_P
    np_P, mp_P = P, mp.mpf(P)

    m0_np = fsolve(np_func, m0_guess)[0]
    m0_mp = mp.findroot(mp_func, [0.98*m0_np, 1.02*m0_np], solver='bisect', tol=1e-10)
    return mp.mpf(m0_mp)


def fourier_mag2(msol, P):
    out = []
    period = mp.mpf(P)

    def m0_integrand(t):
      return msol(t)

    m0 = mp.quad(m0_integrand, [0, period]) / period
    out.append(m0**2)

    for N in range(1, num_k):
      fac = 2/period
      cos_int = mp.quad(lambda t: msol(t)*mp.cos(2*mp.pi*N*t/period), [0, period]) * fac
      sin_int = mp.quad(lambda t: msol(t)*mp.sin(2*mp.pi*N*t/period), [0, period]) * fac
      mn_complex = 0.5*(cos_int - 1j*sin_int)
      out.append(mp.re(mn_complex*mn_complex.conjugate()))
    return out

log_eps = np.linspace(-9.0, -7.0, 5)
EPS = 10.0**log_eps
P_LIST  = P_c * (1.0 - EPS)

EPS_MP  = [mp.mpf(e) for e in EPS]
P_LIST_MP = [mp.mpf(P) for P in P_LIST]

M0_PC   = solve_m0(P_c)
SOL_PC  = mp.odefun(mp_tdgl_deriv, mp.mpf(0), M0_PC)
MN2_PC  = fourier_mag2(SOL_PC, mp_P_c)

rows, cols = num_k, len(P_LIST)
mn2 = [[mp.mpf(0) for _ in range(cols)] for _ in range(rows)]
zn2 = [[mp.mpf(0) for _ in range(cols)] for _ in range(rows)]

for i, P in enumerate(P_LIST):
    m0   = solve_m0(P)
    sol  = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
    mn2_i = fourier_mag2(sol, mp.mpf(P))

    for N in range(rows):
        mn2[N][i] = mn2_i[N]
        diff      = mn2_i[N] - MN2_PC[N]
        zn2[N][i] = abs(diff)

plt.figure(figsize=(3.5, 3.5))
plt.title(r"$z_N^2$ below $P_c$")
plt.xlabel(r"$\varepsilon = (P_c-P)/P_c$")
plt.ylabel(r"$z_N^2$")
plt.xscale('log'); plt.yscale('log')
plt.plot(EPS_MP, EPS_MP, 'k', lw=1, label='bla')
for N in range(rows):
    plt.plot(EPS_MP, [float(z) for z in zn2[N]], label=f'y')
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
