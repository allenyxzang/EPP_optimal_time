import numpy as np
import matplotlib.pyplot as plt

import time
_start = time.perf_counter()


# parameters
F0 = 0.95  # raw fidelity
t1 = 0.01  # latency when the second pair is generated (in 1/gamma)
t2 = 0.1  # wait time after first pair is generated until utilization (in 1/gamma)

# T = 1.0
N_ab = 100
N_t  = 500


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


# approximation of transition border
def border_loc(fid):
    
    loc = (8 * fid**2 - 4 * fid + 5) / (20 * fid**2 - 4 * fid + 2)
    
    return loc


# final fidelity after EPP at t
def fid(a, b, t):
    
    relative_strengths = [a, b, 1-a-b]
    
    bds1 = BDS(diag_elem=[0.95, 0.05/3, 0.05/3, 0.05/3], pauli_strength=relative_strengths)
    bds2 = BDS(diag_elem=[0.95, 0.05/3, 0.05/3, 0.05/3], pauli_strength=relative_strengths)
    
    bds1.evolve_pauli(t)
    bds2.evolve_pauli(t - t1)
    
    diag_elem_new = DEJMPS_res(bds1, bds2)
    
    bds_new = BDS(diag_elem=diag_elem_new[0], pauli_strength=relative_strengths)
    bds_new.evolve_pauli(t2 - t)
    
    fid = bds_new.diag_elem[0]
    
    return fid

# build grids
a = np.linspace(0,1,N_ab)
b = np.linspace(0,1,N_ab)
t = np.linspace(t1,t2,N_t)

A, B = np.meshgrid(a, b, indexing='xy')

# mask off the invalid triangle
mask = (A + B) <= 1.0

# broadcast A, B to shape (Nb, Na, Nt)
A3 = A[:,:,None]        # shape (N_b, N_a, 1)
B3 = B[:,:,None]        # shape (N_b, N_a, 1)
t3 = t[None,None,:]     # shape (1, 1, N_t)

# evaluate f on the full (a,b,t) grid in one shot:
F = fid(A3, B3, t3)       # ⇒ shape (N_b, N_a, N_t)

# find the index of the max along the t-axis:
idx = np.nanargmax(np.where(mask[:,:,None], F, -np.inf), axis=2)

# map indices back to t values
T_opt = t[idx]          # shape (N_b, N_a)
# T_opt = np.where(mask, t[idx], None)

_end = time.perf_counter()
print(f"Total runtime: {_end - _start:.3f} s")

# 1) build a masked array: mask out the invalid region
T_opt_masked = np.ma.array(T_opt, mask=~mask)

# 2) grab your base colormap and set the “bad” color
cmap = plt.get_cmap('bwr').copy()
cmap.set_bad(color='white')    # or any CSS color you like

fig, ax = plt.subplots(figsize=(6,5), dpi=1000)

pcm = ax.pcolormesh(A, B, T_opt_masked, cmap=cmap, vmin=t1, vmax=t2)

# create the colorbar, capture the object
cbar = fig.colorbar(pcm, ax=ax, label=r'Optimal $t$')

# 1) change the colorbar’s axis‐label size
cbar.set_label(r'Optimal $t$', fontsize=18)

# 2) change the tick‐label size on the colorbar
cbar.ax.tick_params(labelsize=16)


# 1a) Axis-label sizes
ax.set_xlabel('$a$', fontsize=18)    # x-axis label
ax.set_ylabel('$b$', fontsize=18)    # y-axis label

# 1b) Tick-label sizes
ax.tick_params(axis='both', which='major', labelsize=16)

loc = border_loc(F0)
a_line = np.linspace(0, loc, 200)
b_line = loc - a_line
ax.plot(a_line, b_line, '--', color='orange', lw=3)

# keep triangle boundary
ax.set_aspect('equal')


plt.tight_layout()
plt.show()

_end_plot = time.perf_counter()
print(f"Total plotting runtime: {_end_plot - _end:.3f} s")

