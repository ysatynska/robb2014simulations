import numpy as np
from scipy.integrate import solve_ivp

def get_diff_fun(P, Omega, h, T):
  fun = {}
  for m_not in np.arange(-1, 1.01, .01):
    def dm_dxi(xi, m):
      return (-m + np.tanh((m + h * np.cos(xi))/T))/Omega
    sol = solve_ivp(dm_dxi, (0, P), [m_not], t_eval=np.arange(0, P, .001))
    diff = sol.y[0][0] - sol.y[0][-1]
    fun[diff] = m_not
  return fun

def get_m0_vals (number, P, Omega, h, T):
  diff_fun = get_diff_fun(P, Omega, h, T)
  target = 0.0
  sorted_keys = sorted(diff_fun.keys(), key=lambda k: abs(k - target))
  closest_keys = sorted_keys[:number]
  closest_vals = [diff_fun[k] for k in closest_keys]
  return closest_vals