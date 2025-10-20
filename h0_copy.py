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

P_c = 5.31935766199729
P_c_mp = mp.mpf(P_c)

h0s = np.logspace(-4, -7, num=2)
z_k_all = [[mp.mpf(0) for _ in h0s] for _ in range(num_k)]

def get_init_m(period, h0=0, m0_guess=-.55, m0_prev=None):
    """Return the P-periodic fixed-point magnetisation at drive offset h0.

    Uses a continuation strategy:
    • If we just solved a nearby h0, start from that solution.
    • Otherwise fall back to a Newton/Brent stage to bracket the root,
      then finish with a high-precision bisection.
    """
    np_P, mp_P = period, mp.mpf(period)

    # ── TDGL RHS (NumPy & mpmath) ──────────────────────────────────────────────
    def np_tdgl_deriv(m, t):
        return -2*np_a*m - 4*np_b*m**3 + np_h1*np.cos(2*np.pi*t/np_P) + h0

    def mp_tdgl_deriv(t, m):
        return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp_P) + h0

    # ── helper: fixed-point residue with mpmath ODE solver ─────────────────────
    def residue(m0):
        sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
        return sol(mp_P) - m0

    # 1) choose a starting guess ------------------------------------------------
    if m0_prev is not None:               # continuation from previous h0
        seed = m0_prev
    else:                                 # fall back to quick NumPy stage
        def np_func(m0):
            msol = odeint(np_tdgl_deriv, m0, np.linspace(0, np_P, num_divs))
            return msol[-1] - m0
        seed = fsolve(np_func, m0_guess, xtol=1e-6)[0]     # looser tol → speed

    # 2) bracket the root so that f(a)*f(b) < 0 --------------------------------
    bracket = [0.9*seed, 1.1*seed]
    for j in range(12):                   # expand at most 12× if necessary
        f1, f2 = residue(bracket[0]), residue(bracket[1])
        if f1*f2 < 0:                     # sign change achieved
            break
        # widen the interval symmetrically
        hw = 0.5*(bracket[1]-bracket[0])
        bracket = [bracket[0]-hw, bracket[1]+hw]
    else:
        raise RuntimeError("Could not bracket root for h0 = %g" % h0)

    # 3) high-precision bisection (guaranteed to converge once bracketed) ------
    with mp.workdps(30):                  # temporarily raise precision
        return mp.findroot(residue, bracket, solver='bisect', tol=1e-12)


def get_m_k(m0, period, h0=0):
    def mp_tdgl_deriv(t, m):
      return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/mp.mpf(period)) + h0
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

for j, h0 in enumerate(h0s):
  init_m_c = get_init_m(P_c, h0)

  m_k_2 = get_m_k(init_m_c, mp.mpf(P_c), h0)

  for k in range(num_k):
    diff = abs(m_k_2[k] - m_k_c_2[k])
    print(diff)
    z_k_all[k][j] = diff

plt.figure(figsize=(3.5, 3.5))
plt.title(r"$z_k$ va $\varepsilon$")
plt.xlabel("h0")
plt.ylabel(r"$z_k$")
plt.xscale('log'); plt.yscale('log')

plt.plot(h0s, [mp.power(i,mp.mpf(1.0)/mp.mpf(3.0)) for i in h0s], 'k--', lw=1, label=r"$\varepsilon$")
for k in range(num_k):
    plt.plot(h0s, [float(z) for z in z_k_all[k]], label=fr"$z_{{{k}}}$")
plt.legend(fontsize=7)
plt.tight_layout()
plt.show()
