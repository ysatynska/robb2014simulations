# import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import itertools

# Set number of digits for mpmath arbitrary precision and disable warnings
mp.dps = 20
np.seterr(all='ignore') # solve_ivp can throw harmless warnings

# --- Global Parameters ---
# Using mpmath types directly for consistency
mp_a = -mp.mpf(3) * mp.sqrt(mp.mpf(3)) / mp.mpf(4)
mp_b = mp.mpf(3) * mp.sqrt(mp.mpf(3)) / mp.mpf(8)
mp_h1 = mp.mpf(1.5)
P_c = mp.mpf('5.31935766199729')

# Perturbation field coefficients
mp_h0 = mp.mpf('0.3')
mp_h2 = mp.mpf('0.4')
mp_h4 = mp.mpf('0.2')

# Number of fourier components to calculate
NUM_FOURIER_COMPS = 4

# --- Function Definitions ---

def get_initial_m(period, h_mult=mp.mpf(0)):
    """
    Finds the initial magnetization m(0) for a periodic solution.
    It first finds an approximate root using numpy/scipy and then refines it
    with mpmath for high precision.
    """
    p_np = float(period)
    h_mult_np = float(h_mult)
    h0_np, h2_np, h4_np = float(mp_h0), float(mp_h2), float(mp_h4)
    a_np, b_np, h1_np = float(mp_a), float(mp_b), float(mp_h1)

    # 1. Numpy/Scipy for a good initial guess
    def np_tdgl_deriv(t, m):
        return -2*a_np*m - 4*b_np*m**3 + h1_np*np.cos(2*np.pi*t/p_np) + \
               h_mult_np * (h0_np + h2_np*np.cos(2*2*np.pi*t/p_np) + h4_np*np.cos(4*2*np.pi*t/p_np))

    def np_residue(m0):
        sol = solve_ivp(np_tdgl_deriv, (0, p_np), [m0], dense_output=True, atol=1e-9, rtol=1e-9)
        return sol.sol(p_np)[0] - m0

    # Find the approximate root
    m0_guess = -0.55
    m0_soln_np = root_scalar(np_residue, x0=m0_guess, xtol=1.0e-10, method='secant')
    
    # 2. MPMath for high-precision root
    def mp_tdgl_deriv(t, m):
        return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/period) + \
               h_mult * (mp_h0 + mp_h2*mp.cos(2*2*mp.pi*t/period) + mp_h4*mp.cos(4*2*mp.pi*t/period))

    def mp_residue(m0):
        sol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
        return sol(period) - m0

    # Refine the root using the numpy solution as a center for the bracket
    m0_seed = mp.mpf(m0_soln_np.root)
    m0_soln_mp = mp.findroot(mp_residue, [m0_seed * 0.95, m0_seed * 1.05], tol=1e-15, solver='bisect')
    
    return m0_soln_mp

def get_fourier_coeffs(m0, period, h_mult=mp.mpf(0), num_comps=4):
    """
    Calculates the complex Fourier coefficients for a given solution.
    """
    def mp_tdgl_deriv(t, m):
        return -2*mp_a*m - 4*mp_b*m**3 + mp_h1*mp.cos(2*mp.pi*t/period) + \
               h_mult * (mp_h0 + mp_h2*mp.cos(2*2*mp.pi*t/period) + mp_h4*mp.cos(4*2*mp.pi*t/period))

    # Get the ODE solution function
    msol = mp.odefun(mp_tdgl_deriv, mp.mpf(0), m0)
    
    mn_coeffs = []
    
    # n = 0 component (the mean)
    m0_integrand = lambda t: msol(t)
    m0_val = mp.quad(m0_integrand, [0, period]) / period
    mn_coeffs.append(m0_val + 0j)

    # n > 0 components
    for n in range(1, num_comps):
        # Note: mn_complex is 0.5 * (mn_cos - 1j * mn_sin) where mn_cos/sin integrals have a 2/P factor.
        # This is equivalent to (1/P) * integral(m(t) * exp(-i*n*w*t) dt)
        integrand_cos = lambda t: msol(t) * mp.cos(2*mp.pi*n*t/period)
        integrand_sin = lambda t: msol(t) * mp.sin(2*mp.pi*n*t/period)
        
        mn_cos = (2 / period) * mp.quad(integrand_cos, [0, period])
        mn_sin = (2 / period) * mp.quad(integrand_sin, [0, period])
        
        mn_complex = 0.5 * (mn_cos - 1j * mn_sin)
        mn_coeffs.append(mn_complex)
        
    return np.array(mn_coeffs, dtype=object)

# --- Main Execution ---

# 1. Calculate the critical solution (h_mult = 0)
print("Calculating critical solution at P=Pc, h_mult=0...")
m0_c = get_initial_m(P_c, h_mult=mp.mpf(0))
mnc_array = get_fourier_coeffs(m0_c, P_c, h_mult=mp.mpf(0), num_comps=NUM_FOURIER_COMPS)
print("Critical Fourier components (mnc_array):\n", mnc_array)

# 2. Calculate solutions for a range of h_mult values
h_mult_list = np.logspace(-11, -10, 3)
mn_results = []

print("\nCalculating solutions for various h_mult...")
for h_mult in h_mult_list:
    h_mult_mp = mp.mpf(h_mult)
    print(f"h_mult = {h_mult_mp}")
    m0_h = get_initial_m(P_c, h_mult=h_mult_mp)
    mn_h = get_fourier_coeffs(m0_h, P_c, h_mult=h_mult_mp, num_comps=NUM_FOURIER_COMPS)
    mn_results.append(mn_h)

# 3. Calculate the differences (delta_m)
# delta_m[i, n] corresponds to h_mult_list[i] and Fourier component n
delta_m = np.array([res - mnc_array for res in mn_results], dtype=object)

# --- Plotting ---

marker_cycler = itertools.cycle(['o','^','v','s','*','+','D'])
import itertools
import numpy as np
import matplotlib.pyplot as plt

h_ref = list(h_mult_list)
eps_floor = 1e-300  # floor to allow log-scale when a component is exactly zero

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))  # <-- no sharey / sharex

# =========================
# Left panel: EVEN n modes
# =========================
ax = axes[0]
ax.set_title(r"Scaling of $|\delta m_n|$ (even $n$)", fontsize=15, fontweight="bold")
ax.set_xlabel(r"$h_{\mathrm{mult}}$", fontsize=22)
ax.set_ylabel(r"$|\delta m_n|$", fontsize=22)
ax.set_xscale('log'); ax.set_yscale('log')

# Reference: h^{1/3}
ref_even = [float(mp.power(h, mp.mpf(1)/3)) for h in h_ref]
ax.plot(h_ref, ref_even, 'k--', lw=1, label=r"$h_{\mathrm{mult}}^{1/3}$")

marker_even = itertools.cycle(['o','^','v','s','*','+','D'])
for n in range(0, NUM_FOURIER_COMPS, 2):
    re_vals = [float(abs(mp.re(d[n]))) for d in delta_m]
    im_vals = [float(abs(mp.im(d[n]))) for d in delta_m]
    re_vals = np.maximum(re_vals, eps_floor)
    im_vals = np.maximum(im_vals, eps_floor)
    m = next(marker_even)
    ax.plot(h_ref, re_vals, marker=m, linestyle='-',  label=fr"$|\Re(\delta m_{{{n}}})|$")
    if n > 0:
        ax.plot(h_ref, im_vals, marker=m, linestyle='--', label=fr"$|\Im(\delta m_{{{n}}})|$")

ax.legend(loc="upper left", fontsize=13, frameon=True)

# =======================
# Right panel: ODD n modes
# =======================
ax = axes[1]
ax.set_title(r"Scaling of $|\delta m_n|$ (odd $n$)", fontsize=15, fontweight="bold")
ax.set_xlabel(r"$h_{\mathrm{mult}}$", fontsize=22)
ax.set_xscale('log'); ax.set_yscale('log')

# Reference: h^{2/3}
ref_odd = [float(mp.power(h, mp.mpf(2)/3)) for h in h_ref]
ax.plot(h_ref, ref_odd, 'k--', lw=1, label=r"$h_{\mathrm{mult}}^{2/3}$")

marker_odd = itertools.cycle(['o','^','v','s','*','+','D'])
for n in range(1, NUM_FOURIER_COMPS, 2):
    re_vals = [float(abs(mp.re(d[n]))) for d in delta_m]
    im_vals = [float(abs(mp.im(d[n]))) for d in delta_m]
    re_vals = np.maximum(re_vals, eps_floor)
    im_vals = np.maximum(im_vals, eps_floor)
    m = next(marker_odd)
    ax.plot(h_ref, re_vals, marker=m, linestyle='-',  label=fr"$|\Re(\delta m_{{{n}}})|$")
    ax.plot(h_ref, im_vals, marker=m, linestyle='--', label=fr"$|\Im(\delta m_{{{n}}})|$")

ax.legend(loc="upper left", fontsize=13, frameon=True)

fig.tight_layout()
plt.show()
