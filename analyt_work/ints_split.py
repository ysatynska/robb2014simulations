import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy import integrate

mp.dps = 25
mp_a = -mp.mpf(3) * mp.sqrt(3) / 4
mp_b = mp.mpf(3) * mp.sqrt(3) / 8
mp_h1 = mp.mpf('1.5')
P_c_mp = mp.mpf('5.31935766199729')
P = mp.mpf('5.3193577')
h_mults = np.logspace(-14, -5, 5)

mp_h0 = mp.mpf('0.3')
mp_h2 = mp.mpf('0.4')
mp_h4 = mp.mpf('0.2')

n_range = range(-32, 32 + 1)

def get_init_m(period, h_mult=mp.mpf(0)):
    p_np = float(period)
    h_mult_np = float(h_mult)
    a_np, b_np, h1_np = float(mp_a), float(mp_b), float(mp_h1)
    h0_np, h2_np, h4_np = float(mp_h0), float(mp_h2), float(mp_h4)

    def np_tdgl_deriv(t, m):
        return -2*a_np*m - 4*b_np*m**3 + h1_np*np.cos(2*np.pi*t/p_np) + \
               h_mult_np * (h0_np + h2_np*np.cos(2*2*np.pi*t/p_np) + h4_np*np.cos(4*2*np.pi*t/p_np))

    def np_residue(m0):
        sol = solve_ivp(np_tdgl_deriv, (0, p_np), [m0], dense_output=True, atol=1e-9, rtol=1e-9)
        return sol.sol(p_np)[0] - m0

    m0_soln_np = root_scalar(np_residue, x0=-0.55, xtol=1.0e-10, method='secant')

    def mp_tdgl_deriv(t, m):
        return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*np.pi*t/period) + \
               h_mult * (mp_h0 + mp_h2*mp.cos(2*2*np.pi*t/period) + mp_h4*mp.cos(4*2*np.pi*t/period))

    def mp_residue(m0):
        sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
        return sol(period) - m0

    m0_seed = mp.mpf(m0_soln_np.root)
    m0_soln_mp = mp.findroot(mp_residue, [m0_seed * 0.95, m0_seed * 1.05], tol=1e-20, solver='bisect')
    return m0_soln_mp

def get_m_k(m0, period, h_mult=mp.mpf(0), num_k=32):
  def mp_tdgl_deriv(t, m):
    return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*np.pi*t/period) + h_mult * (mp_h0 + mp_h2*mp.cos(2*2*np.pi*t/period) + mp_h4*mp.cos(4*2*np.pi*t/period))
  
  sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
  m_k_cs = []
  
  m0_mean = mp.quad(lambda t: sol(t), [0, period]) / period
  m_k_cs.append(m0_mean)

  for k in range(-num_k, num_k+1):
    omega_k = 2 * np.pi * k / period
    cos_int = mp.quad(lambda t: sol(t) * mp.cos(omega_k * t), [0, period]) / period
    sin_int = mp.quad(lambda t: sol(t) * mp.sin(omega_k * t), [0, period]) / period
    m_k_cs.append(cos_int - 1j * sin_int)
    
  return np.array(m_k_cs)

init_m_c = get_init_m(P_c_mp)
# mk_c = get_m_k(init_m_c, P_c_mp)

def get_m_fun(m0, period, h_mult=mp.mpf(0)):
  def mp_tdgl_deriv(t, m):
    return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/period) + \
            h_mult * (mp_h0 + mp_h2*mp.cos(2*2*mp.pi*t/period) + mp_h4*mp.cos(4*2*mp.pi*t/period))
  return mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)

m_c_fun = get_m_fun(init_m_c, P_c_mp, h_mult=0)

I1_vals = []
I1_second_part = []
I2_vals = []
I3_vals = []
I4_vals = []
actual_sum = []
should_be_sum = []

for h_mult in h_mults:
  print(h_mult)
  m0_P = get_init_m(P, h_mult)
  m_fun = get_m_fun(m0_P, P, h_mult)
  dm = lambda t: m_fun(t) - m_c_fun(t) 
  dm_even = lambda t: 0.5*(dm(t) + dm(t + P/2))
  dm_odd = lambda t: 0.5*(dm(t) - dm(t + P/2))

  def int1_integrand(t):
    m = m_c_fun(t)
    return 2 * mp_a * dm_even(t)
  int1 = integrate.quad(int1_integrand, 0, P)[0]
  print("int1", int1)
  I1_vals.append(abs(int1))

  def i1_second_integrand(t):
    m = m_c_fun(t)
    return 12 * mp_b * m**2 * dm_even(t)
  i1_second = integrate.quad(i1_second_integrand, 0, P)[0]
  print("i1_second", i1_second)
  I1_second_part.append(abs(i1_second))

  def int2_integrand(t):
    return dm_even(t)**3
  int2 = 4 * mp_b * integrate.quad(int2_integrand, 0, P)[0]
  I2_vals.append(abs(int2))

  def int3_integrand(t):
    m = m_c_fun(t)
    return m * dm_odd(t) * dm_even(t)
  int3 = 24 * mp_b * integrate.quad(int3_integrand, 0, P)[0]
  I3_vals.append(abs(int3))

  def int4_integrand(t):
    return dm_odd(t) * dm_odd(t) * dm_even(t)
  int4 = 12 * mp_b * integrate.quad(int4_integrand, 0, P)[0]
  I4_vals.append(abs(int4))

  def should_be_sum_integrand(t):
    return dm_even(t)
  sb_sum = integrate.quad(should_be_sum_integrand, 0, P)[0]
  should_be_sum.append(sb_sum)

  actual_sum.append(int1 + int2 + int3 + int4 + i1_second)


plt.loglog(h_mults, I1_vals, label=r'$|2ad_me|$')
# plt.loglog(h_mults, I1_second_part, label=r'$|12bm0**2dm|$')
plt.loglog(h_mults, I2_vals, label=r'$|I_2|$')
plt.loglog(h_mults, I3_vals, label=r'$|I_3|$')
plt.loglog(h_mults, I4_vals, label=r'$|I_4|$')
plt.loglog(h_mults, actual_sum, label="Actual-sum")
# plt.loglog(h_mults, should_be_sum, label="Should-be-sum")
plt.loglog(h_mults, h_mults, 'k--', lw=1, label=r'$h_{mult}$', color="black")
plt.loglog(h_mults, h_mults**(1/3), lw=1, label=r'$h_0^{1/3}$', color="black")
plt.xlabel(r'$h_{mults}$'); plt.ylabel('scaling terms')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout(); plt.show()