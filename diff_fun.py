from scipy import integrate
from scipy.integrate import solve_ivp
import math
import numpy as np
import matplotlib.pyplot as plt

h1 = 1.5 # 
P = 7 # period
max_time = 1500
transient_t = 1000
step = .01
time_step = .01
m_avgs = []

a = -3 * math.sqrt(3)/4 # material-specific constant
b = 3 * math.sqrt(3)/8 # material-specific constant

t_vals = np.arange(0, max_time, time_step)
# h = h1 * np.cos(2 * np.pi * t_vals / P)


fun = {}
for m_not in np.arange(-1, 1, step):
    def dm_dt(t, m):
        return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / P)
    sol = solve_ivp(dm_dt, (0, max_time), [m_not], t_eval=t_vals, dense_output=True)
    diff = sol.sol(m_not)[0] - sol.sol(m_not + P)[0]
    fun[diff] = m_not

target = 0.0

# find the key closest to target
closest_key = min(fun.keys(), key=lambda k: abs(k - target))

# grab its associated value
closest_value = fun[closest_key]
print(closest_value)

m0 = [closest_value]

# Is = []
# for i in np.arange(4, P, step):
#     def dm_dt(t, m):
#         return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / i)
#     sol = solve_ivp(dm_dt, (0, max_time), m0, t_eval=t_vals, dense_output=True)

#     def fun_to_integrate(t):
#         m = sol.sol(t)[0]
#         return 2 * a + 12 * b * m * m
    
#     def get_m_fun(t):
#         return sol.sol(t)[0]
    
#     result = integrate.quad(fun_to_integrate, 0, i)[0]
#     m_avg = integrate.quad(get_m_fun, 0, i)[0]

#     print("finished", i)
#     Is.append([i, result/i])
#     m_avgs.append(m_avg/i)

plt.plot(sorted(fun.keys()), [fun[k] for k in sorted(fun.keys())])
# plt.plot([x[0] for x in Is], [x for x in m_avgs])
# plt.title(f'Max Time - {max_time}; Step - {step}; M_0 - {m0}')
plt.xlabel("Period P")
plt.ylabel("Integral I")
plt.show()