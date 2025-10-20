import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

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

def get_init_m(period, h_mult=mp.mpf(0)):
    p_np = float(period)
    h_mult_np = float(h_mult)
    a_np, b_np, h1_np = float(mp_a), float(mp_b), float(mp_h1)
    h0_np, h2_np, h4_np = float(mp_h0), float(mp_h2), float(mp_h4)
    def np_tdgl_deriv(t, m):
        return -2*a_np*m - 4*b_np*m**3 + h1_np*np.cos(2*np.pi*t/p_np) + h_mult_np*(h0_np + h2_np*np.cos(4*np.pi*t/p_np) + h4_np*np.cos(8*np.pi*t/p_np))
    def np_residue(m0):
        sol = solve_ivp(np_tdgl_deriv, (0, p_np), [m0], dense_output=True, atol=1e-9, rtol=1e-9)
        return sol.sol(p_np)[0] - m0
    m0_soln_np = root_scalar(np_residue, x0=-0.55, xtol=1.0e-10, method='secant')
    def mp_tdgl_deriv(t, m):
        return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/period) + h_mult*(mp_h0 + mp_h2*mp.cos(4*mp.pi*t/period) + mp_h4*mp.cos(8*mp.pi*t/period))
    def mp_residue(m0):
        sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
        return sol(period) - m0
    m0_seed = mp.mpf(m0_soln_np.root)
    try:
        m0_soln_mp = mp.findroot(mp_residue, (m0_seed*mp.mpf('0.95'), m0_seed*mp.mpf('1.05')), tol=mp.mpf('1e-28'))
    except:
        m0_soln_mp = mp.findroot(mp_residue, (m0_seed*mp.mpf('0.8'), m0_seed*mp.mpf('1.2')), solver='bisect', tol=mp.mpf('1e-28'))
    return m0_soln_mp

def get_m_fun(m0, period, h_mult=mp.mpf(0)):
    def mp_tdgl_deriv(t, m):
        return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/period) + h_mult*(mp_h0 + mp_h2*mp.cos(4*mp.pi*t/period) + mp_h4*mp.cos(8*mp.pi*t/period))
    return mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)

init_m_c = get_init_m(P_c_mp)
m_c_fun = get_m_fun(init_m_c, P_c_mp, h_mult=0)

I1_vals = []
I2_vals = []
I3_vals = []
I4_vals = []
actual_sum = []
# should_be_sum = []

for h_mult in h_mults:
    print(h_mult)
    h_mult_mp = mp.mpf(str(h_mult))
    m0_P = get_init_m(P, h_mult_mp)
    m_fun = get_m_fun(m0_P, P, h_mult_mp)

    P_half = P/2
    def dm(t): return m_fun(t) - m_c_fun(t)
    def dm_even(t): return (dm(t) + dm(t + P_half))/2
    def dm_odd(t):  return (dm(t) - dm(t + P_half))/2
    def A_of_t(t):
        m = m_c_fun(t)
        return (2*mp_a + 12*mp_b*m*m)

    extra = max(0, int(-mp.log10(h_mult_mp))) + 6
    with mp.workdps(int(mp.dps) + extra):
        int1 = mp.quad(lambda t: A_of_t(t)*dm_even(t), [0, P])
        int2 = 4*mp_b * mp.quad(lambda t: dm_even(t)**3, [0, P])
        int3 = 24*mp_b * mp.quad(lambda t: m_c_fun(t)*dm_odd(t)*dm_even(t), [0, P])
        int4 = 12*mp_b * mp.quad(lambda t: dm_odd(t)*dm_odd(t)*dm_even(t), [0, P])
        sb_sum = mp.quad(lambda t: dm_even(t), [0, P])

    I1_vals.append(abs(int1))
    I2_vals.append(abs(int2))
    I3_vals.append(abs(int3))
    I4_vals.append(abs(int4))
    actual_sum.append(int1 + int2 + int3 + int4)
    # should_be_sum.append(sb_sum)

plt.loglog(h_mults, I1_vals, label=r'$|I_1|$')
plt.loglog(h_mults, I2_vals, label=r'$|I_2|$')
plt.loglog(h_mults, I3_vals, label=r'$|I_3|$')
plt.loglog(h_mults, actual_sum, label="Actual-sum")
# plt.loglog(h_mults, should_be_sum, label="Should-be-sum")
plt.loglog(h_mults, h_mults, '--', linewidth=1, label=r'$h_{mult}$')
plt.xlabel(r'$h_{mult}$'); plt.ylabel('scaling terms')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout(); plt.show()
