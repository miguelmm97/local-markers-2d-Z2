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
loger_marker = logging.getLogger('marker')
loger_marker.setLevel(logging.INFO)

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
loger_marker.addHandler(stream_handler)



#%% Local marker ED
def bulk_avg_marker(site_pos, local_marker, Nx, Ny, cutoff_x=0.25, cutoff_y=0.25):
    bool_x_right = site_pos[:, 0] < (Nx - 1) * (1 - cutoff_x)
    bool_x_left = site_pos[:, 0] > (Nx - 1) * cutoff_x
    bool_y_right = site_pos[:, 1] < (Ny - 1) * (1 - cutoff_y)
    bool_y_left = site_pos[:, 1] > (Ny - 1) * cutoff_y
    bool_site = bool_x_right & bool_x_left & bool_y_right & bool_y_left
    return np.sum(local_marker[bool_site]) / np.sum(bool_site)

def local_marker(x, y, S_tilde, Nx=None, Ny=None):
    """
    Input:
    x, y -> np.ndarray: coordinates of the lattice sites
    P -> np.ndarray: One particle density matrix (projector onto the filled bands)
    S_tilde -> np.ndarray: auxiliary "spin" operator
    Nx -> int: number of sites along x
    Ny -> int: number of sites along y

    Output:
    local_marker -> np.ndarray: local z2 marker at each site
    """

    # Calculation of the local marker
    if Nx is None or Ny is None:
        loger_marker.error('To shift sites Nx and Ny must be specified')
    X, Y = np.repeat(x - 0.5 * (Nx - 1), 4), np.repeat(y - 0.5 * (Ny - 1), 4)
    X = np.reshape(X, (len(X), 1))
    Y = np.reshape(Y, (len(Y), 1))
    XS = X * S_tilde
    YS = Y * S_tilde
    M = S_tilde @ XS @ YS - S_tilde @ YS @ XS

    local_marker = np.zeros((len(x),))
    for i in range(len(x)):
        loger_marker.trace(f'Calculating marker at site: {i}/{len(x)-1}')
        idx = 4 * i
        local_marker[i] = - (pi / 8) * np.trace(np.imag(M[idx: idx + 4, idx: idx + 4]))

        # Reality check (diag M should have no real part, it is antihermitian)
        if loger_marker.isEnabledFor(logging.DEBUG):
            if np.allclose(1e-13, np.real(np.diag(M[idx: idx + 4, idx: idx + 4]))):
                loger_marker.debug('Local marker real: True')
            else:
                loger_marker.warning(f'Local marker has a non vanishing '
                                     f'imaginary part: {np.max(np.abs(np.real(np.diag(M[idx: idx + 4]))))}')
    return local_marker

def local_marker_old(x, y, rho, S, spin_mixing=False, shift_per_site=False, Nx=None, Ny=None):
    """
    Input:
    x, y -> np.ndarray: coordinates of the lattice sites
    P -> np.ndarray: One particle density matrix (projector onto the filled bands)
    S -> np.ndarray: onsite auxiliary (kronecker product of onsite chiral symmetry)
    Spin-mixing -> bool: whether spin is conserved or not
    shift_per_site -> bool: Whether the marker is calculated at each site by shifting its position to the centre
    Nx -> int: number of sites along x
    Ny -> int: number of sites along y

    Output:
    local_marker -> np.ndarray: local z2 marker at each site
    """

    if spin_mixing:
        # vals, vecs = eigh(S @ rho + rho @ S - S)
        # PS = vecs @ np.diag(np.sign(vals)) @ vecs.T.conj()
        # C = - 1 / (32 * pi)
        I = np.eye(rho.shape[0], dtype=np.complex128)
        Q = I - 2 * rho
        Qp = 2 * (Q + S @ Q @ S)
        vals, vecs = eigh(Qp)
        Qp = vecs @ np.diag(np.sign(vals)) @ vecs.T.conj()
        rho = (I - Qp) / 2
        PS = rho @ S
        C = pi
    else:
        PS = rho @ S
        C = pi

    local_marker = np.zeros((len(x),))
    if shift_per_site:
        if Nx is None or Ny is None:
            loger_kwant.error('To shift sites Nx and Ny must be specified')
        for i in range(len(x)):
            loger_kwant.trace(f'Calculating marker at site: {i}/{len(x)-1}')
            X, Y = np.repeat(x - 0.5 * (Nx - 1), 4), np.repeat(y - 0.5 * (Ny - 1), 4)
            X = np.reshape(X, (len(X), 1))
            Y = np.reshape(Y, (len(Y), 1))
            XPS = X * PS
            YPS = Y * PS
            M = PS @ XPS @ YPS - PS @ YPS @ XPS
            idx = 4 * i
            local_marker[i] = C * np.trace(np.imag(M[idx: idx + 4, idx: idx + 4]))
    else:
        X, Y = np.repeat(x, 4), np.repeat(y, 4)
        X = np.reshape(X, (len(X), 1))
        Y = np.reshape(Y, (len(Y), 1))
        XPS = X * PS
        YPS = Y * PS
        M = PS @ XPS @ YPS - PS @ YPS @ XPS
        for i in range(len(x)):
            idx = 4 * i
            local_marker[i] = C * np.trace(np.imag(M[idx: idx + 4, idx: idx + 4]))

    return local_marker
