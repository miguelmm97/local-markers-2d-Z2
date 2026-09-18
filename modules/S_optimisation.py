# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.integrate import quad
import scipy.linalg as la
from functools import partial
from scipy.sparse import diags
from scipy.linalg import sqrtm
from numpy.linalg import eigh, eigvalsh

# Kwant
import kwant
import tinyarray as ta
from kwant.kpm import jackson_kernel

# Managing logging
import logging

loger_S_opt = logging.getLogger(__name__)


#%% Pauli matrices

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z


#%% Auxiliary "spin" operator for the Rashba-BHZ model

def S_tilde_linear_basis(rho, dim_Hext):
    """
    Input:
    rho -> np.ndarray: One particle density matrix (projector onto the filled bands)
    dim_Hext -> int: dimension of the external hilbert space

    Output:
    S_tilde_0 -> np.ndarray: raw (un-flattened) S_tilde at theta = 0
    S_tilde_1 -> np.ndarray: raw (un-flattened) S_tilde at theta = pi / 2
    """

    S0 = np.kron(np.eye(dim_Hext), np.kron(tau_z, sigma_z))
    S1 = np.kron(np.eye(dim_Hext), np.kron(tau_y, sigma_0))
    return rho @ S0 + S0 @ rho - S0, rho @ S1 + S1 @ rho - S1


def S_tilde_gap(S_tilde_0, S_tilde_1, theta):
    """
    Gap of S_tilde at one angle, without ever building the flattened operator.
    Meant for the theta scan, which only reads the gap

    Input:
    S_tilde_0, S_tilde_1 -> np.ndarray: the pair returned by S_tilde_linear_basis
    theta -> float: angle parametrising the one-parameter family of "spin" operators

    Output:
    gap -> float: closest the raw spectrum of S_tilde gets to zero
    """

    return np.min(np.abs(eigvalsh(np.cos(theta) * S_tilde_0 + np.sin(theta) * S_tilde_1)))


def get_S_tilde_and_gap(rho, theta, dim_Hext, zero_mode_tol=1e-8):
    """
    Input:
    rho -> np.ndarray: One particle density matrix (projector onto the filled bands)
    theta -> float: angle parametrising the one-parameter family of "spin" operators
    dim_Hext -> int: dimension of the external hilbert space
    zero_mode_tol -> float: below this distance to zero an eigenvalue counts as a zero
                            mode and the flattening is reported as ill defined

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
    gap = np.min(np.abs(vals))

    # Zero modes warning
    n_exact_zero = int(np.count_nonzero(vals == 0.))
    n_zero_mode = int(np.count_nonzero(np.abs(vals) <= zero_mode_tol))
    if n_exact_zero > 0:
        loger_S_opt.warning(f'{n_exact_zero} eigenvalue(s) of S_tilde vanish exactly at '
                            f'theta={theta:.4f}. They are dropped by the sign flattening, so '
                            f'S_tilde ** 2 != I and any marker built on it is meaningless.')
    elif n_zero_mode > 0:
        loger_S_opt.warning(f'{n_zero_mode} eigenvalue(s) of S_tilde lie within {zero_mode_tol:.1e} '
                            f'of zero at theta={theta:.4f} (gap={gap:.3e}). Their sign is fixed by '
                            f'numerical noise, so the flattening is ambiguous at this angle.')

    # Flattening: S_tilde -> sign(S_tilde), so that S_tilde ** 2 = I
    S_tilde = vecs @ np.diag(np.sign(vals)) @ vecs.T.conj()

    # The raw spectrum is returned as well so that the diagnostics below do not need to
    # diagonalise S_tilde a second time
    return S_tilde, gap, vals
