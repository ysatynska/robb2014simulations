import math
import mpmath
import numpy as np
from scipy import integrate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from get_m0_vals_ising import get_m0_vals
import cmath
from scipy.integrate import quad

num_k = 6

h1 = 1.5
num_m0s = 1
max_time = 100
time_step = .001
t_vals = np.arange(0, max_time, time_step)
a = -3 * math.sqrt(3)/4 # material-specific constant
b = 3 * math.sqrt(3)/8 # material-specific constant
P_c = 5.31935766199729

def get_diff_fun (P):
  fun = {}
  for init_m in np.arange(-1.5, 1.51, .01):
    def dm_dt(t, m):
      return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / P)
    sol = solve_ivp(dm_dt, (0, P), [init_m], t_eval=np.arange(0, P, .001))
    diff = sol.y[0][0] - sol.y[0][-1]
    fun[diff] = init_m
  return fun

def get_init_m (P):
  diff_fun = get_diff_fun(P)
  closest_key = sorted(diff_fun, key=abs)[0]
  return diff_fun[closest_key]

init_m_Pc = get_init_m(P_c)
print("init_m_Pc: ", init_m_Pc)



t_evals_Pc = np.arange(0, P_c, 0.001)
def dm_dt(t, m):
  return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / P_c)
sol_c = solve_ivp(dm_dt, (0, P_c), [init_m_Pc], t_eval=t_evals_Pc, dense_output=True)
m_vals = sol_c.y[0]

# My original method
m_k_c = []
for k in range(num_k):
  def integrand_real(t):
    return np.real(sol_c.sol(t)[0] * np.exp(-1j * k * 2 * np.pi * t / P_c))
  def integrand_imag(t):
    return np.imag(sol_c.sol(t)[0] * np.exp(-1j * k * 2 * np.pi * t / P_c))
  real_part, _ = quad(integrand_real, 0, P_c)
  imag_part, _ = quad(integrand_imag, 0, P_c)
  m_k_val = (real_part + 1j * imag_part) / P_c
  m_k_c.append(m_k_val)
m_k_c = np.array(m_k_c)
print("Original Integration by parts:", m_k_c, '\n')

omega = 2 * np.pi / P_c

# Trapezoidal Rule
m_k_c = []
for k in range(num_k):
    exp_term = np.exp(-1j * k * omega * t_evals_Pc)
    integrand = m_vals * exp_term
    integral = np.trapz(integrand, t_evals_Pc) / P_c
    m_k_c.append(integral)
m_k_c = np.array(m_k_c)
print("Trapezoidal rule:", m_k_c, '\n')

# FFT
dt = t_evals_Pc[1] - t_evals_Pc[0]
fft_vals = np.fft.fft(m_vals) / len(m_vals)
frequencies = np.fft.fftfreq(len(m_vals), d=dt)
m_k_c = fft_vals[:num_k]
print("FFT:", m_k_c, '\n')

# Euler's formula decomposition
m_k_c = []
for k in range(num_k):
    cos_term = np.cos(k * omega * t_evals_Pc)
    sin_term = np.sin(k * omega * t_evals_Pc)

    real_part = np.trapz(m_vals * cos_term, t_evals_Pc)
    imag_part = np.trapz(m_vals * sin_term, t_evals_Pc)

    m_k = (real_part - 1j * imag_part) / P_c
    m_k_c.append(m_k)
m_k_c = np.array(m_k_c)
print("Euler's decomposition:", m_k_c, '\n')


P_vals = np.linspace(P_c - 0.00001, P_c - 0.000000001, 30)
P_vals = [5.31918944913828, 5.31882572623109]
z_k_all = []
epsilons = []
for P in P_vals:
  epsilons.append((P_c - P) / P_c)
  init_m = get_init_m(P)
  print(P, init_m)
  t_evals_P = np.arange(0, 10 * P, .001)
  def dm_dt(t, m):
    return -2 * a * m - 4 * b * m**3 + h1 * np.cos(2 * np.pi * t / P)
  sol = solve_ivp(dm_dt, (0, 10 * P), [init_m], t_eval=t_evals_P, dense_output=True)

  m_k = []
  for k in range(num_k):
    def integrand_real(t):
      return np.real(sol_c.sol(t)[0] * np.exp(-1j * k * 2 * np.pi * t / P))
    def integrand_imag(t):
      return np.imag(sol_c.sol(t)[0] * np.exp(-1j * k * 2 * np.pi * t / P))
    real_part, _ = quad(integrand_real, 0, P)
    imag_part, _ = quad(integrand_imag, 0, P)
    m_k_val = (real_part + 1j * imag_part) / P
    m_k.append(m_k_val)
  m_k = np.array(m_k)

  diff = np.real(m_k * m_k.conj()) - np.real(m_k_c * m_k_c.conj())
  print(diff)
  z_k = np.sqrt(diff)

  z_k_all.append(z_k)

z_k_all = np.array(z_k_all)
epsilons = np.array(epsilons)

for k in range(num_k):
  plt.loglog(epsilons, z_k_all[:, k], label=f'z_{k}')
plt.loglog(epsilons, epsilons**0.5, 'k--', label='ε^1/2')
plt.xlabel("ε")
plt.ylabel("z_k")
plt.title("Critical scaling: $z_k$ vs ε")
plt.legend()
plt.grid(True, which='both', ls=':')
plt.show()