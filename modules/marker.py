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

loger_marker = logging.getLogger(__name__)


# %% Local marker ED
def bulk_avg_marker(site_pos, local_marker, Nx, Ny, cutoff_x=0.25, cutoff_y=0.25):
    bool_x_right = site_pos[:, 0] < (Nx - 1) * (1 - cutoff_x)
    bool_x_left = site_pos[:, 0] > (Nx - 1) * cutoff_x
    bool_y_right = site_pos[:, 1] < (Ny - 1) * (1 - cutoff_y)
    bool_y_left = site_pos[:, 1] > (Ny - 1) * cutoff_y
    bool_site = bool_x_right & bool_x_left & bool_y_right & bool_y_left
    return np.sum(local_marker[bool_site]) / np.sum(bool_site)

def local_marker_OBC_naive(x, y, S_tilde, Nx=None, Ny=None):
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
        loger_marker.trace(f'Calculating marker at site: {i}/{len(x) - 1}')
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

def marker_per_site_speedup(x, y, S_tilde, Nx=None, Ny=None, boundary='Open'):
    """
    Local z2 marker with every site evaluated in its own coordinate frame, so that each
    site sits as far as possible from the branch cut of the position operator.

    The frames are precomputed as the minimum-image relative coordinates and packed into
    two rescaled copies of S_tilde, so all sites come out of a single matrix product
    instead of one triple product per site. This requires S_tilde to be hermitian, which
    is what lets the two orderings of the commutator be collapsed into one term below.

    For a detailed derivation see the notes "marker_pbc"

    Input:
    x, y -> np.ndarray: coordinates of the lattice sites
    S_tilde -> np.ndarray: flattened auxiliary "spin" operator
    Nx -> int: number of sites along x (only needed for closed boundaries)
    Ny -> int: number of sites along y (only needed for closed boundaries)
    boundary -> str: 'Open' or 'Closed', matching AmorphousLattice_2d.boundary

    Output:
    marker -> np.ndarray: local z2 marker at each site
    """

    if boundary not in ('Open', 'Closed'):
        raise ValueError(f"boundary must be 'Open' or 'Closed', got {boundary!r}")
    if boundary == 'Closed' and (Nx is None or Ny is None):
        raise ValueError("Nx and Ny must be specified for the minimum image under boundary='Closed'")

    N = len(x)
    loger_marker.trace(f'Calculating the marker at {N} sites, {boundary} boundaries...')

    # Row i holds the positions of every site as seen from site i
    Dx = x[None, :] - x[:, None]
    Dy = y[None, :] - y[:, None]
    if boundary == 'Closed':
        Dx = (Dx + Nx / 2) % Nx - Nx / 2
        Dy = (Dy + Ny / 2) % Ny - Ny / 2


    S4 = S_tilde.reshape(N, 4, N, 4)
    P = (S4 * Dx[:, None, :, None]).reshape(4 * N, 4 * N)
    Q = S4 * Dy.T[:, None, :, None]
    R = (P @ S_tilde).reshape(N, 4, N, 4)
    trace_per_site = np.einsum('ipbq,bqip->i', R, Q)

    # Reality check (the diagonal blocks are antihermitian, so their trace is imaginary)
    if loger_marker.isEnabledFor(logging.DEBUG):
        real_residue = np.abs(np.real(trace_per_site)).max()
        if real_residue < 1e-10:
            loger_marker.debug('Local marker real: True')
        else:
            loger_marker.warning(f'Local marker has a non vanishing real part: {real_residue}')

    return - (pi / 4) * np.imag(trace_per_site)

