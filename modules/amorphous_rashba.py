# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.integrate import quad
import scipy.linalg as la
from functools import partial
from scipy.sparse import diags
from scipy.linalg import sqrtm
from numpy.linalg import eigh

# Kwant
import kwant
import tinyarray as ta
from kwant.kpm import jackson_kernel

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Classes
from modules.AmorphousLattice_2d import AmorphousLattice_2d_Kwant

# %% Logging setup
loger_kwant = logging.getLogger('kwant')
loger_kwant.setLevel(logging.DEBUG)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'black',
        'DEBUG': 'purple',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
stream_handler.setFormatter(formatter)
loger_kwant.addHandler(stream_handler)

#%% Pauli matrices

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z


#%% Rashba-type SOC Hamiltonian

def displacement2D_kwant(site0, site1):
    x1, y1 = site0.pos[0], site0.pos[1]
    x2, y2 = site1.pos[0], site1.pos[1]

    v = np.zeros((2,))
    v[0] = (x2 - x1)
    v[1] = (y2 - y1)

    # Norm of the vector between sites 2 and 1
    r = np.sqrt(v[0] ** 2 + v[1] ** 2)

    # Phi angle of the vector between sites 2 and 1 (angle in the XY plane)
    if v[0] == 0:                                    # Pathological case, separated to not divide by 0
        if v[1] > 0:
            phi = pi / 2                             # Hopping in y
        else:
            phi = 3 * pi / 2                         # Hopping in -y
    else:
        if v[1] > 0:
            phi = np.arctan2(v[1], v[0])             # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])    # 3rd and 4th quadrants

    return r, phi

def hopping(A, lambR, d, phi, cutoff_dist):

    # radial factor
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)

    # inter-orbital hopping
    hopp_orb = A * np.kron(tau_z, sigma_0)

    # spin-orbit + rashba
    s_par = np.cos(phi) * sigma_x + np.sin(phi) * sigma_y
    s_perp = np.sin(phi) * sigma_x - np.cos(phi) * sigma_y
    hopp_SOC = - 1j * A * np.kron(tau_x, s_par)
    hopp_rashba = -1j * lambR * np.kron(tau_0, s_perp)

    return f_cutoff * (hopp_orb + hopp_SOC + hopp_rashba)

def rashba_syst_Kwant(lattice_tree, param_dict):

    # Load parameters into the builder namespace
    try:
        M = param_dict['M']
        W = param_dict['W']
        A = param_dict['A']
        lambR = param_dict['lambR']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create SiteFamily from the amorphous lattice
    latt = AmorphousLattice_2d_Kwant(norbs=4, lattice=lattice_tree, name='RASHBA')

    # Onsite mass and Anderson disorder
    def onsite_potential(site):
        index = site.tag[0]
        if lattice_tree.K_onsite < 1e-12:
            return M * np.kron(tau_z, sigma_0)
        else:
            return M * np.kron(tau_z, sigma_0) + lattice_tree.onsite_disorder[index] * np.eye(4)

    # Hopping
    def hopp(site1, site0):
        d, phi = displacement2D_kwant(site1, site0)
        return hopping(A, lambR, d, phi, lattice_tree.r)

    # Initialise kwant system
    loger_kwant.trace('Creating kwant scattering region...')
    syst = kwant.Builder()
    syst[(latt(i) for i in range(latt.Nsites))] = onsite_potential

    # Populate hoppings
    for i in range(latt.Nsites):
        for n in lattice_tree.neighbours[i]:
            loger_kwant.trace(f'Defining hopping from site {i} to {n}.')
            syst[(latt(n), latt(i))] = hopp

    return syst

