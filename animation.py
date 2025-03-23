import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

h1 = 1.5
max_time = 50
time_step = 1
m0 = [1.1, -0.9]

a = -3 * math.sqrt(3) / 4
b = 3 * math.sqrt(3) / 8

t_vals = np.arange(0, max_time, time_step)

fig, axs = plt.subplots(3, 1, figsize=(6, 10))
line_m, = axs[0].plot([], [], label="m(t)")
line_h, = axs[1].plot([], [], color='orange', label="h(t)")
line_hyst1, = axs[2].plot([], [], color='red', label="m1(t) vs h(t)")
line_hyst2, = axs[2].plot([], [], color='blue', label="m2(t) vs h(t)")
# Pause/resume
is_paused = {'value': False}
def on_click(event):
    if is_paused['value']:
        ani.event_source.start()
        is_paused['value'] = False
    else:
        ani.event_source.stop()
        is_paused['value'] = True
fig.canvas.mpl_connect('button_press_event', on_click)

for ax, ylab in zip(axs, ["m(t)", "h(t)", "m(t)", "m(t)"]):
  ax.set_xlim(-h1 - 0.1, h1 + 0.1)
  ax.set_ylim(-1.5, 1.5)
  ax.set_ylabel(ylab)
  ax.grid(True)

axs[2].set_xlabel("h(t)")

def update(frame):
  P = frame
  h = h1 * np.cos(2 * np.pi * t_vals / P)

  def dm_dt(t, m):
      return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / P)
  sol = solve_ivp(dm_dt, (0, max_time), m0, t_eval=t_vals)

  line_m.set_data(t_vals, sol.y[0])
  line_h.set_data(t_vals, h)

  line_hyst1.set_data(h, sol.y[0])
  line_hyst2.set_data(h, sol.y[1])

  axs[0].set_xlim(0, 2 * P)
  axs[1].set_xlim(0, 2 * P)

  fig.suptitle(f"Current Period P = {P}", fontsize=14)
  return line_m, line_h, line_hyst1, line_hyst2

ani = animation.FuncAnimation(
  fig, update, frames=np.arange(6, 5, -.01), interval=100, blit=False
)

plt.tight_layout()
plt.show()

