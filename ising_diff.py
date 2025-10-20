import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Omega = .2 * np.pi
h = .55
T = .45
# m0 = [.8, -.8]
max_time = 20 * np.pi
time_step = .001
t_vals = np.arange(0, max_time, time_step)

fun = {}
for m_not in np.arange(-1, 1.01, .01):
  def dm_dxi(xi, m):
    return (-m + np.tanh((m + h * np.cos(xi))/T))/Omega
  sol = solve_ivp(dm_dxi, (0, 2 * np.pi), [m_not], t_eval=np.arange(0, 2 * np.pi, time_step))
  diff = sol.y[0][0] - sol.y[0][-1]
  fun[diff] = m_not

target = 0.0
sorted_keys = sorted(fun.keys(), key=lambda k: abs(k - target))
closest_keys = sorted_keys[:3]
closest_vals = [fun[k] for k in closest_keys]
m0 = closest_vals
print(m0)

# def dm_dxi(xi, m):
#   return (-m + np.tanh((m + h * np.cos(xi))/T))/Omega

# sol = solve_ivp(dm_dxi, (0, max_time), m0, t_eval=t_vals)
# print(sol)

# plt.figure()
# plt.plot(sol.t, sol.y[0], label='m(ξ)')
# plt.plot(sol.t, h * np.cos(sol.t), '--', label='h cosξ')
# plt.plot(sol.t, sol.y[1], label='m(ξ)')

# plt.plot(sol.t, sol.y[2], label='m(ξ)')
# plt.plot(sol.t, sol.y[3], label='m(ξ)')
# plt.plot(sol.t, sol.y[4], label='m(ξ)')

# plt.plot(sol.t, sol.y[5], label='m(ξ)')
# plt.plot(sol.t, sol.y[6], label='m(ξ)')
# plt.plot(sol.t, sol.y[7], label='m(ξ)')

# plt.plot(sol.t, sol.y[8], label='m(ξ)')
# plt.plot(sol.t, sol.y[9], label='m(ξ)')
# plt.ylim(-1, 1)
# # plt.xlim(60 * np.pi, 62 * np.pi)
# plt.xlabel("Xi")
# plt.ylabel("m")
# plt.legend()
# plt.show()