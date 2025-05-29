import numpy as np
import matplotlib.pyplot as plt

import time
_start = time.perf_counter()


F0=0.95


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


def EN(f):
    """Entanglement (logarithmic) negativity of BDS"""
    EN = 1 + np.log2(f)
    
    return EN


def f(t1, t2, t):
    """
    Example function. Replace with your own f(t1, t2, t).
    Must accept numpy arrays for t and return array of same shape.
    """
    # Here we use a simple illustrative example:
    # a peaked function in the interval [t1, t2]
    return -((t - (t1 + t2)/2)**2) + 0.1*(t2 - t1)


def fid(t1, t2, t):
    
    relative_strengths = [1/3, 1/3, 1/3]
    
    bds1 = BDS(diag_elem=[F0, (1-F0)/3, (1-F0)/3, (1-F0)/3], pauli_strength=relative_strengths)
    bds2 = BDS(diag_elem=[F0, (1-F0)/3, (1-F0)/3, (1-F0)/3], pauli_strength=relative_strengths)
    
    bds1.evolve_pauli(t)
    bds2.evolve_pauli(t - t1)
    
    diag_elem_new = DEJMPS_res(bds1, bds2)
    p_succ = diag_elem_new[1]
    
    bds_new = BDS(diag_elem=diag_elem_new[0], pauli_strength=relative_strengths)
    bds_new.evolve_pauli(t2 - t)
    
    fid = bds_new.diag_elem[0]
    
    EN_norm = EN(fid) * p_succ

    return EN_norm


# --- Grid settings ---
n_grid = 200  # number of points along each axis
t1_vals = np.linspace(0, 0.1, n_grid)
t2_vals = np.linspace(0, 0.4, n_grid)
Delta = np.full((n_grid, n_grid), np.nan)

# --- Brute‐force search for each (t1, t2) ---
n_t = 500  # number of samples in [t1, t2] for maximization
for i, t1 in enumerate(t1_vals):
    for j, t2 in enumerate(t2_vals):
        if t2 < t1:
            continue
        ts = np.linspace(t1, t2, n_t)
        f_vals = fid(t1, t2, ts)
        idx = np.nanargmax(f_vals)
        Delta[j, i] = ts[idx] - t1

# --- Mask out invalid region (t2 < t1) ---
Delta_masked = np.ma.masked_invalid(Delta)

_end = time.perf_counter()
print(f"Total runtime: {_end - _start:.3f} s")

# --- Plotting ---
# T1, T2 = np.meshgrid(t1_vals, t2_vals, indexing='xy')
# # note: Delta_masked.T so that Δ[i,j] → T1[i,j]=t1_vals[j], T2[i,j]=t2_vals[i]
# plt.figure(figsize=(6,5),dpi=1000)
# plt.pcolormesh(
#     T1, T2, Delta_masked.T,
#     cmap='viridis', shading='auto',
#     vmin=0, vmax=Delta_masked.max()
# )
# plt.xlabel('$t_1$')
# plt.ylabel('$t_2$')
# cbar = plt.colorbar()
# cbar.set_label(r'$\Delta t$')
# plt.tight_layout()
# plt.show()


fig, ax = plt.subplots(figsize=(6, 5),dpi=1000)
cmap = plt.get_cmap('Reds').copy()
cmap.set_bad(color='white')              # color for masked (invalid) region
vmin, vmax = 0, Delta_masked.max()       # fix endpoints of the colormap
norm = plt.Normalize(vmin=vmin, vmax=vmax)

im = ax.imshow(
    Delta_masked,
    origin='lower',
    extent=[0, 0.1, 0, 0.4],
    aspect='auto',
    cmap=cmap,
    norm=norm
)

ax.set_xlabel(r'$t_1$', fontsize=16)
ax.set_ylabel(r'$t_2$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$\Delta t$', fontsize=16)
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()

_end_plot = time.perf_counter()
print(f"Total plotting runtime: {_end_plot - _end:.3f} s")
