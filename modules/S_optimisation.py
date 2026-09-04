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

# %% Logging setup
loger_S_opt = logging.getLogger('S_opt')
loger_S_opt.setLevel(logging.DEBUG)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'white',
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
loger_S_opt.addHandler(stream_handler)


#%% Pauli matrices

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z


#%% Auxiliary "spin" operator for the Rashba-BHZ model

def rashba_bhz_S_tilde(rho, theta, dim_Hext):
    """
    Input:
    rho -> np.ndarray: One particle density matrix (projector onto the filled bands)
    theta -> float: angle parametrising the one-parameter family of "spin" operators
    dim_Hext -> int: dimension of the external hilbert space

    Output:
    S_tilde -> np.ndarray: flattened auxiliary "spin" operator (S_tilde ** 2 = I)
    gap -> float: closest the raw spectrum of S_tilde gets to zero
    vals -> np.ndarray: raw (un-flattened) spectrum of S_tilde, sorted in ascending order
    """

    # One-parameter family of onsite "spin" operators
    S0 = np.kron(tau_z, sigma_z)
    S_theta = np.kron(np.eye(dim_Hext), np.cos(theta) * S0 + np.sin(theta) * np.kron(tau_y, sigma_0))

    # Auxiliary operator and its raw spectrum
    S_tilde = rho @ S_theta + S_theta @ rho - S_theta
    vals, vecs = eigh(S_tilde)

    # The gap is the distance of the raw spectrum to zero. It sets how well defined the
    # flattening below is: an eigenvalue sitting at zero makes the sign ambiguous
    gap = np.min(np.abs(vals))

    # Flattening: S_tilde -> sign(S_tilde), so that S_tilde ** 2 = I
    S_tilde = vecs @ np.diag(np.sign(vals)) @ vecs.T.conj()

    # The raw spectrum is returned as well so that the diagnostics below do not need to
    # diagonalise S_tilde a second time
    return S_tilde, gap, vals


def local_DoS(state, Nsites):
    local_DoS = np.zeros((Nsites, ), dtype=np.complex128)
    for i in range(Nsites):
        psi_i = state[i * 4: i * 4 + 4]
        local_DoS[i] = psi_i.T.conj() @ psi_i

    if np.sum(np.imag(local_DoS)) < 1e-10:
        local_DoS = np.real(local_DoS)
    else:
        raise TypeError('DoS is complex.')

    return local_DoS