import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Omega = .2 * np.pi
h = .55
T = .39
m0 = [.8, -.8]
max_time = 20 * np.pi
time_step = .001
t_vals = np.arange(0, max_time, time_step)

def dm_dxi(xi, m):
  return (-m + np.tanh((m + h * np.cos(xi))/T))/Omega

sol = solve_ivp(dm_dxi, (0, max_time), m0, t_eval=t_vals)

plt.figure()
plt.plot(sol.t, sol.y[0], label='m(ξ)')
plt.plot(sol.t, h * np.cos(sol.t), '--', label='h cosξ')
plt.plot(sol.t, sol.y[1], label='m(ξ)')
plt.ylim(-1, 1)
# plt.xlim(6 * np.pi, 8 * np.pi)
plt.xlabel("Xi")
plt.ylabel("m")
plt.legend()
plt.show()


