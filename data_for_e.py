from scipy import integrate
from scipy.integrate import solve_ivp
import math
# fsolve
import numpy as np
import matplotlib.pyplot as plt

h1 = 1.5 # 
P = 6 # period
Pc = 5.319357661995
max_time = 15000
transient_t = 10000
step = .0001
time_step = .01
m0 = [1.2]

a = -3 * math.sqrt(3)/4 # material-specific constant
b = 3 * math.sqrt(3)/8 # material-specific constant

t_vals = np.arange(0, max_time, time_step)
# h = h1 * np.cos(2 * np.pi * t_vals / P)

m_avgs = []
es = []

for i in np.arange(5.29, Pc, step):
    def dm_dt(t, m):
        return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / i)
    sol = solve_ivp(dm_dt, (0, max_time), m0, t_eval=t_vals, dense_output=True)
    
    def get_m_fun(t):
        return sol.sol(t)[0]
    
    m_avg = integrate.quad(get_m_fun, transient_t, transient_t + i)[0]
    m_avgs.append(m_avg/i)
    print("finished", i)
    e = (Pc - i) / Pc
    es.append(e)
print(es)
offset = 1e-8
# es_shifted = [x + abs(min(es)) + offset for x in es]
m_avgs_shifted = [x + abs(min(m_avgs)) + offset for x in m_avgs]

fig, axs = plt.subplots(2, 1, figsize=(5, 10))

# ε vs ⟨m⟩
axs[0].plot([x for x in es], [x for x in m_avgs_shifted], color="blue")
axs[0].set_xlabel(r"$\varepsilon$")
axs[0].set_ylabel(r"$\langle m \rangle$")
axs[0].set_title(
    rf"Order Parameter vs. Control Parameter "
    rf"(Transient = {transient_t}, Max Time = {max_time}, Step = {step})",
    fontsize=12, fontweight="bold"
)

# log(ε) vs ⟨m⟩
axs[1].plot([math.log(x) for x in es], [x for x in m_avgs_shifted], color="orange")
axs[1].set_xlabel(r"$\log(\varepsilon)$")
axs[1].set_ylabel(r"$\langle m \rangle$")
axs[1].set_title(
    r"Log-Scaled Control Parameter vs. Order Parameter",
    fontsize=12, fontweight="bold"
)

plt.tight_layout()
plt.show()
