import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root_scalar
from scipy.integrate import odeint

mp.dps = 25
mp_a = -mp.mpf(3) * mp.sqrt(3) / 4
mp_b = mp.mpf(3) * mp.sqrt(3) / 8
mp_h1 = mp.mpf('1.5')
np_a = -3.0*np.sqrt(3.0)/4.0
np_b = 3.0*np.sqrt(3.0)/8.0
np_h1 = 1.5
P_c_mp = mp.mpf('5.31935766199729')
num_k = 4
h_mults = np.logspace(-11, -10, 5)

mp_h0 = mp.mpf('0.3')
mp_h2 = mp.mpf('0.4')
mp_h4 = mp.mpf('0.2')

np_h0 = 0.3
np_h2 = 0.4
np_h4 = 0.2


num_divs = 10000

def get_init_m(period, h_mult=mp.mpf(0)):
    p_np = float(period)
    h_mult_np = float(h_mult)
    m0_guess=-0.55

    def np_tdgl_deriv(t, m):
        return -2*np_a*m - 4*np_b*m**3 + np_h1*np.cos(2*np.pi*t/p_np) + \
               h_mult_np * (np_h0 + np_h2*np.cos(2*2*np.pi*t/p_np) + np_h4*np.cos(4*2*np.pi*t/p_np))

    def mp_tdgl_deriv(t, m):
        return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*np.pi*t/period) + \
               h_mult * (mp_h0 + mp_h2*mp.cos(2*2*np.pi*t/period) + mp_h4*mp.cos(4*2*np.pi*t/period))

    def np_func(m0):
      msol = odeint(np_tdgl_deriv, m0, np.linspace(0, p_np, num_divs))
      return msol[-1] - m0

    m0_seed = fsolve(np_func, m0_guess, xtol=1e-6)[0]

    def residue(mp_m0):
      sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), mp_m0)
      return sol(period) - mp_m0

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

def get_m_k(m0, period, h_mult=mp.mpf(0)):
    def mp_tdgl_deriv(t, m):
      return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*np.pi*t/period) + h_mult * (mp_h0 + mp_h2*mp.cos(2*2*np.pi*t/period) + mp_h4*mp.cos(4*2*np.pi*t/period))
    
    sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
    m_k_cs = []
    
    m0_mean = mp.quad(lambda t: sol(t), [0, period]) / period
    m_k_cs.append(m0_mean)

    for k in range(1, num_k):
      omega_k = 2 * np.pi * k / period
      cos_int = mp.quad(lambda t: sol(t) * mp.cos(omega_k * t), [0, period]) / period
      sin_int = mp.quad(lambda t: sol(t) * mp.sin(omega_k * t), [0, period]) / period
      m_k_cs.append(cos_int - 1j * sin_int)
      
    return np.array(m_k_cs)

init_m_c = get_init_m(P_c_mp)
mk_c = get_m_k(init_m_c, P_c_mp)

delta_mks = []
for h_mult in h_mults:
    print(h_mult)
    h_mult_mp = mp.mpf(h_mult)
    m0_h = get_init_m(P_c_mp, h_mult_mp)
    mk = get_m_k(m0_h, P_c_mp, h_mult_mp)
    delta_mks.append(mk - mk_c)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].set_title(r"$dm_n$ (n even) vs. $h_{mult}$")
axs[0].plot(h_mults, [mp.power(i, mp.mpf(1)/3) for i in h_mults], 'k--', lw=2, label=r"$h_{mult}^{1/3}$")
axs[0].set_xscale('log'); axs[0].set_yscale('log')
for n in range(0, num_k, 2):
  axs[0].plot(h_mults, [abs(mp.re(d[n])) for d in delta_mks], linestyle='-', label=rf'$|Re(dm_{n})|$')
  if n > 0:
    axs[0].plot(h_mults, [abs(mp.im(d[n])) for d in delta_mks], linestyle='--', label=rf'$|Im(dm_{n})|$')
axs[0].set_xlabel(r"$h_{mult}$"); axs[0].set_ylabel(r"$|dm_n|$"); axs[0].legend()

axs[1].set_title(r"$dm_n$ (n odd) vs. $h_{mult}$")
axs[1].plot(h_mults, [mp.power(i, mp.mpf(2)/3) for i in h_mults], 'k--', lw=2, label=r"$h_{mult}^{2/3}$")
axs[1].set_xscale('log'); axs[1].set_yscale('log')
for n in range(1, num_k, 2):
    axs[1].plot(h_mults, [abs(mp.re(d[n])) for d in delta_mks], linestyle='-', label=rf'$|Re(dm_{n})|$')
    axs[1].plot(h_mults, [abs(mp.im(d[n])) for d in delta_mks], linestyle='--', label=rf'$|Im(dm_{n})|$')
axs[1].set_xlabel(r"$h_{mult}$"); axs[1].set_ylabel(r"$|dm_n|$"); axs[1].legend()

plt.tight_layout()
plt.show()