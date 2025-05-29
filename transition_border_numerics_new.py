import numpy as np
import matplotlib.pyplot as plt

import time
_start = time.perf_counter()


"""Define time-dependent coefficients for Pauli channel evolution."""
def pI(t, pauli_strength):
    coeff = (1 + np.exp(- 2 * (pauli_strength[0] + pauli_strength[1]) * t) + \
             np.exp(- 2 * (pauli_strength[2] + pauli_strength[1]) * t) + \
                 np.exp(- 2 * (pauli_strength[0] + pauli_strength[2]) * t)) / 4
    
    return coeff

def pX(t, pauli_strength):
    coeff = (1 - np.exp(- 2 * (pauli_strength[0] + pauli_strength[1]) * t) + \
             np.exp(- 2 * (pauli_strength[2] + pauli_strength[1]) * t) - \
                 np.exp(- 2 * (pauli_strength[0] + pauli_strength[2]) * t)) / 4
    
    return coeff

def pY(t, pauli_strength):
    coeff = (1 - np.exp(- 2 * (pauli_strength[0] + pauli_strength[1]) * t) - \
             np.exp(- 2 * (pauli_strength[2] + pauli_strength[1]) * t) + \
                 np.exp(- 2 * (pauli_strength[0] + pauli_strength[2]) * t)) / 4
    
    return coeff

def pZ(t, pauli_strength):
    coeff = (1 + np.exp(- 2 * (pauli_strength[0] + pauli_strength[1]) * t) - \
             np.exp(- 2 * (pauli_strength[2] + pauli_strength[1]) * t) - \
                 np.exp(- 2 * (pauli_strength[0] + pauli_strength[2]) * t)) / 4
    
    return coeff

def CI(t, pauli_strength):
    coeff = ((pI(t, pauli_strength)) ** 2 + (pX(t, pauli_strength)) ** 2 + \
             (pY(t, pauli_strength)) ** 2 + (pZ(t, pauli_strength)) ** 2)
    
    return coeff

def CX(t, pauli_strength):
    coeff = (2 * pI(t, pauli_strength) * pX(t, pauli_strength) + \
             2 * pY(t, pauli_strength) * pZ(t, pauli_strength))
    
    return coeff

def CY(t, pauli_strength):
    coeff = (2 * pI(t, pauli_strength) * pY(t, pauli_strength) + \
             2 * pX(t, pauli_strength) * pZ(t, pauli_strength))
    
    return coeff

def CZ(t, pauli_strength):
    coeff = (2 * pI(t, pauli_strength) * pZ(t, pauli_strength) + \
             2 * pY(t, pauli_strength) * pX(t, pauli_strength))
    
    return coeff


class BDS():
    """Class of Bell diagonal states, tracking 4 diagonal elements.
    
    Have evolution methods to undergo error channels.
    """
    def __init__(self, diag_elem=[1,0,0,0], pauli_strength=[1/3,1/3,1/3]):
    # def __init__(self, diag_elem, pauli_strength):
        assert len(diag_elem)==4, "Need 4 diagonal elements."
        assert len(pauli_strength)==3, "Need 3 Pauli components."
        # assert abs(sum(diag_elem)-1)<1e-3, "Diagonal elements should sum to 1."
        # assert abs(sum(pauli_strength)-1)<1e-3, "Relative strength should be normalized."
        
        # Bell diagonal elements, in order of psi+, psi-, phi+, phi-
        self.diag_elem = diag_elem
        # relative strength of three Pauli channels, in order of x, y, z
        self.pauli_strength = pauli_strength
        
    def evolve_pauli(self, t):
        """Method that modifies diagonal elements according to Pauli channel time evolution."""
        lambda1 = (self.diag_elem[0] * CI(t, self.pauli_strength) + self.diag_elem[1] * CZ(t, self.pauli_strength) + \
                   self.diag_elem[2] * CX(t, self.pauli_strength) + self.diag_elem[3] * CY(t, self.pauli_strength))
        lambda2 = (self.diag_elem[0] * CZ(t, self.pauli_strength) + self.diag_elem[1] * CI(t, self.pauli_strength) + \
                   self.diag_elem[2] * CY(t, self.pauli_strength) + self.diag_elem[3] * CX(t, self.pauli_strength))
        lambda3 = (self.diag_elem[0] * CX(t, self.pauli_strength) + self.diag_elem[1] * CY(t, self.pauli_strength) + \
                   self.diag_elem[2] * CI(t, self.pauli_strength) + self.diag_elem[3] * CZ(t, self.pauli_strength))
        lambda4 = (self.diag_elem[0] * CY(t, self.pauli_strength) + self.diag_elem[1] * CX(t, self.pauli_strength) + \
                   self.diag_elem[2] * CZ(t, self.pauli_strength) + self.diag_elem[3] * CI(t, self.pauli_strength))
        
        self.diag_elem[0] = lambda1
        self.diag_elem[1] = lambda2
        self.diag_elem[2] = lambda3
        self.diag_elem[3] = lambda4
        
    def evolve_ad(self, t):
        """Method that modifies diagonal elements according to amplitude damping time evolution."""
        lambda1 = (2 * self.diag_elem[0] * np.exp(-t) + (self.diag_elem[2]+self.diag_elem[3]) * (np.exp(-t) - np.exp(-2*t))) / 2
        lambda2 = (2 * self.diag_elem[1] * np.exp(-t) + (self.diag_elem[2]+self.diag_elem[3]) * (np.exp(-t) - np.exp(-2*t))) / 2
        lambda3 = (self.diag_elem[2] * (1 + np.exp(-2*t)) + self.diag_elem[3] * (1 - np.exp(-t))**2 + \
                   (self.diag_elem[0]+self.diag_elem[1]) * (1 - np.exp(-t))) / 2
        lambda4 = (self.diag_elem[3] * (1 + np.exp(-2*t)) + self.diag_elem[2] * (1 - np.exp(-t))**2 + \
                   (self.diag_elem[0]+self.diag_elem[1]) * (1 - np.exp(-t))) / 2
        
        self.diag_elem[0] = lambda1
        self.diag_elem[1] = lambda2
        self.diag_elem[2] = lambda3
        self.diag_elem[3] = lambda4
        
    def get_fid(self):
        """Method to return the current fidelity (first diagonal element)."""
        
        return self.diag_elem[0]


def DEJMPS_res(bds1: BDS, bds2: BDS):
    """Standard DEJMPS protocol uses CNOT working on BDS with major Phi+.
    
    Therefore for states with major Psi+, conceptually before DEJMPS a Pauli X is needed.
    After purification another Pauli X transforms the major component from Phi+ back to Psi+.
    """
    # first BDS
    lambda1_1 = bds1.diag_elem[0]
    lambda2_1 = bds1.diag_elem[1]
    lambda3_1 = bds1.diag_elem[2]
    lambda4_1 = bds1.diag_elem[3]
    # second BDS
    lambda1_2 = bds2.diag_elem[0]
    lambda2_2 = bds2.diag_elem[1]
    lambda3_2 = bds2.diag_elem[2]
    lambda4_2 = bds2.diag_elem[3]
    
    # success probability
    p_succ = (lambda1_1 + lambda2_1) * (lambda1_2 + lambda2_2) + (lambda3_1 + lambda4_1) * (lambda3_2 + lambda4_2)
    
    lambda1_new = (lambda1_1 * lambda1_2 + lambda2_1 * lambda2_2) / p_succ
    lambda2_new = (lambda1_1 * lambda2_2 + lambda2_1 * lambda1_2) / p_succ
    lambda3_new = (lambda3_1 * lambda3_2 + lambda4_1 * lambda4_2) / p_succ
    lambda4_new = (lambda3_1 * lambda4_2 + lambda4_1 * lambda3_2) / p_succ
    
    res = [[lambda1_new, lambda2_new, lambda3_new, lambda4_new], p_succ]
    
    return res


# --- 1) Define your objective f(F,a,t)
def fid(F, a, t, t1, t2):
    
    relative_strengths = [a, 0, 1-a]
    
    bds1 = BDS(diag_elem=[F, (1-F)/3, (1-F)/3, (1-F)/3], pauli_strength=relative_strengths)
    bds2 = BDS(diag_elem=[F, (1-F)/3, (1-F)/3, (1-F)/3], pauli_strength=relative_strengths)
    
    bds1.evolve_pauli(t)
    bds2.evolve_pauli(t - t1)
    
    diag_elem_new = DEJMPS_res(bds1, bds2)
    
    bds_new = BDS(diag_elem=diag_elem_new[0], pauli_strength=relative_strengths)
    bds_new.evolve_pauli(t2 - t)
    
    fid = bds_new.diag_elem[0]
    
    return fid


# --- 2) Sampling parameters
n_a = 500                  # number of a‐samples in [0,1]
n_F = 20                   # number of F‐samples in [0.5,1]
n_t = 500                  # number of t‐samples per interval
tol = 0.001                  # threshold tolerance for "moved off t1"

a_vals = np.linspace(0, 1, n_a)
F_vals = np.linspace(0.8, 1.0, n_F)

# --- 3) Specify your list of (t1,t2) intervals
#    You can hard‐code or generate them programmatically:
t1_list = np.linspace(0, 0.04, 10)
t2_list = np.linspace(0.06, 0.1, 10)
t_intervals = [(t1, t2) for t1 in t1_list for t2 in t2_list if t2 > t1]

# --- 4) Allocate array to hold thresholds:
#     shape = (#intervals, #F)
thresholds = np.full((len(t_intervals), n_F), np.nan)

# --- 5) Loop over intervals and Fs to find a_th
for i, (t1, t2) in enumerate(t_intervals):
    t_vals = np.linspace(t1, t2, n_t)
    for j, F in enumerate(F_vals):
        a_th = np.nan
        for a in a_vals:
            y = fid(F, a, t_vals, t1, t2)
            t_opt = t_vals[np.argmax(y)]
            if t_opt > t1 + tol:
                a_th = a
                break
        thresholds[i, j] = a_th

# --- 6) Compute statistics across intervals
mean_th = np.nanmean(thresholds, axis=0)
var_th  = np.nanvar(thresholds, axis=0)

# --- 7) Plot mean and variance vs F
# plt.figure(figsize=(8, 6))

# Here we use the raw variance as error bars:
yerr = var_th

# approximation of transition border
def border_loc(fid):
    
    loc = (8 * fid**2 - 4 * fid + 5) / (20 * fid**2 - 4 * fid + 2)
    
    return loc


_end = time.perf_counter()
print(f"Total runtime: {_end - _start:.3f} s")


fig, ax = plt.subplots(figsize=(7,5), dpi=1000)
ax.errorbar(
    F_vals,
    mean_th,
    yerr=yerr,
    fmt='o',
    capsize=4,
    elinewidth=1.5,
    markeredgewidth=1.5,
    label='Numerical'
)

# overlay the analytical curve
ax.plot(
    F_vals,
    border_loc(F_vals),
    '--',
    linewidth=2,
    label='Analytical'
)

# 1a) Axis-label sizes
ax.set_xlabel('$F_0$', fontsize=18)    # x-axis label
ax.set_ylabel('border location', fontsize=18)    # y-axis label

# 1b) Tick-label sizes
xticks = [0.8, 0.85, 0.9, 0.95, 1]
ax.set_xticks(xticks)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

_end_plot = time.perf_counter()
print(f"Total plotting runtime: {_end_plot - _end:.3f} s")
