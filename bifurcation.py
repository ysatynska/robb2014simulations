from sympy import symbols, integrate, solve, Eq
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

h1 = 1.5 # 
P = 1 # period
max_time = 20
time_step = .001
m0 = [1.1, -.9]

a = -3 * math.sqrt(3)/4 # material-specific constant
b = 3 * math.sqrt(3)/8 # material-specific constant

t_vals = np.arange(0, max_time, time_step)
h = h1 * np.cos(2 * np.pi * t_vals / P)

def dm_dt(t, m):
    return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / P)
sol = solve_ivp(dm_dt, (0, max_time), m0, t_eval=t_vals)

plt.figure(figsize=(3.5, 3.5))
# first trajectory: light purple
plt.plot(h, sol.y[0], color="#b19cd9", linewidth=1.8, label=fr"$m(t)$, $m_0={m0[0]}$")
# second trajectory: light blue
plt.plot(h, sol.y[1], color="#87cefa", linewidth=1.8, label=fr"$m(t)$, $m_0={m0[1]}$")

plt.xlabel(r"$h(t)$", fontsize=22)
plt.ylabel(r"$m(t)$", fontsize=22)
plt.title(fr"$m$ vs $h$ for Period $P = 1 = 0.188 P_c$",
          fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(5, 10))
# m(t)
axs[0].plot(sol.t, sol.y[0], color="red")
axs[0].plot(sol.t, sol.y[1], color="blue")
axs[0].set_xlabel("Time t")
axs[0].set_ylabel("m(t)")
axs[0].set_title("m(t)")
axs[0].grid(True)
# h(t)
axs[1].plot(t_vals, h, label="h(t)", color='orange')
axs[1].set_xlabel("Time t")
axs[1].set_ylabel("h(t)")
axs[1].set_title("h(t)")
# h(t) vs m(t)
axs[2].plot(h, sol.y[0], color='red')
axs[2].plot(h, sol.y[1], color='blue')
axs[2].set_xlabel("h(t)")
axs[2].set_ylabel("m(t)")
axs[2].set_title("m(t) vs h(t)")
axs[2].grid(True)
plt.tight_layout()
plt.show()

# for i in range(1, max_time, time_step):
#   mi = m[i - 1] + time_step * (-2 * a * m[i - 1] - 4 * b * m[i - 1] * m[i - 1] * m[i - 1]  + h[i-1])
#   print(f"A {i}, a = {-2 * a * m[i - 1]}")
#   print(f"B {i}, b = {- 4 * b * m[i - 1] * m[i - 1] * m[i - 1]}")
#   print(f"Slope {i}, sl = {-2 * a * m[i - 1] - 4 * b * m[i - 1] * m[i - 1] * m[i - 1]  + h[i-i]}")
#   print(f"Step {i}, h-1 = {h[i - 1]}")
#   print(f"Step {i}, m-1 = {m[i-1]}")
#   print(f"Step {i}, mi = {mi}")
#   m.append(mi)