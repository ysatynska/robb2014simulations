from sympy import symbols, integrate, solve, Eq
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

h1 = 1.5 # 
P = 100 # period
max_time = 200
time_step = 1
m0 = [1.2]

a = -3 * math.sqrt(3)/4 # material-specific constant
b = 3 * math.sqrt(3)/8 # material-specific constant

t_vals = np.arange(0, max_time, time_step)
h = h1 * np.cos(2 * np.pi * t_vals / P)

def dm_dt(t, m):
    return -2 * a * m - 4 * b * m**3 + 1.5 * np.cos(2 * np.pi * t / P)
sol = solve_ivp(dm_dt, (0, max_time), m0, t_eval=t_vals)

fig, axs = plt.subplots(3, 1, figsize=(5, 10))
# m(t)
axs[0].plot(sol.t, sol.y[0], label="m(t)")
axs[0].set_xlabel("Time t")
axs[0].set_ylabel("m(t)")
axs[0].set_title("m(t)")
axs[0].grid(True)
# h(t)
axs[1].plot(t_vals, h, label="h(t)", color='orange')
axs[1].set_xlabel("Time t")
axs[1].set_ylabel("h(t)")
axs[1].set_title("h(t)")
axs[1].grid(True)
# h(t) vs m(t)
axs[2].plot(h, sol.y[0], color='red')
axs[2].set_xlabel("h(t)")
axs[2].set_ylabel("m(t)")
axs[2].set_title("m(t) vs h(t)")
axs[2].grid(True)
plt.tight_layout()
plt.show()