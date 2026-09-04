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

# %% Module

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z



#%% BHZ Hamiltonian
def displacement2D(x1, y1, x2, y2):

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
codex
def hopping(t, lamb, d, phi, cutoff_dist):
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)

    # U(1)-spin hoppings
    hopp_x = np.cos(phi) * (np.kron(tau_z, sigma_0) - 1j * np.kron(tau_x, sigma_z))
    hopp_y = np.sin(phi) * (np.kron(tau_z, sigma_0) - 1j * np.kron(tau_y, sigma_0))

    # U(1)-spin breaking
    mixing_x = 1j * 0.5 * lamb * np.cos(phi) * np.kron(tau_z, sigma_y)
    mixing_y = -1j * 0.5 * lamb * np.sin(phi) * np.kron(tau_z, sigma_x)

    return f_cutoff * (2 * t * (hopp_x + hopp_y) + mixing_x + mixing_y)

def spectrum(H, Nsp=None):
    """
    Input:
    H -> nd.array: Hamiltonian of the nanowire
    Nsp -> int: filling fraction

    Output:
    energy -> np.ndarray: sorted eigenvalues of H
    eigenstates -> np.ndarray: corresponding eigenvectors of H
    P -> np.ndarray: One particle density matrix (projector onto the filled bands)
    """
    if Nsp is None:
        Nsp = int(len(H) / 2)

    # Spectrum
    energy, eigenstates = np.linalg.eigh(H)
    idx = energy.argsort()
    energy = energy[idx]
    eigenstates = eigenstates[:, idx]

    # OPDM
    U = np.zeros((len(H), len(H)), dtype=np.complex128)
    U[:, 0: Nsp] = eigenstates[:, 0: Nsp]
    P = U @ np.conj(np.transpose(U))

    return energy, eigenstates, P


def BHZ_Hamiltonian_Kwant(lattice_tree, param_dict):

    # Load parameters into the builder namespace
    try:
        m = param_dict['m']
        t = param_dict['t']
        lamb = param_dict['lamb']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create SiteFamily from the amorphous lattice
    latt = AmorphousLattice_2d_Kwant(norbs=4, lattice=lattice_tree, name='bbh_model')

    # Hopping and onsite functions
    def onsite_potential(site):
        return m * np.kron(tau_z, sigma_0)

    def hopp(site1, site0):
        d, phi = displacement2D_kwant(site1, site0)
        return hopping(t, lamb, d, phi, lattice_tree.r)

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


