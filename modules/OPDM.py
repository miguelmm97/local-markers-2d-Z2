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

loger_opdm = logging.getLogger(__name__)


#%% OPDM and density

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

def OPDM(eigenvectors, filling=0.5):
    loger_opdm.info('Calculating OPDM...')
    dim = eigenvectors.shape[0]
    Nsp = int(dim * filling)
    U = np.zeros((dim, dim), dtype=np.complex128)
    U[:, 0: Nsp] = eigenvectors[:, 0: Nsp]
    rho = U @ np.conj(np.transpose(U))
    return rho

def reduced_OPDM(rho, site_indices):
    Nred = len(site_indices) * 4
    rho_red = np.zeros((Nred, Nred), dtype=np.complex128)
    for i, site1 in enumerate(site_indices):
        for j, site2 in enumerate(site_indices):
            block1 = site1 * 4
            block2 = site2 * 4
            rho_red[i * 4: i * 4 + 4, j * 4: j * 4 + 4] = rho[block1: block1 + 4, block2: block2 + 4]

    return rho_red

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

def occupied_zero_energy_DoS(rho_eigvecs, H, Nsites, tol=1e-5, filling=0.5):
    zero_energy_Dos = np.zeros((Nsites, ), dtype=np.float64)
    for i in range(int(H.shape[0] * filling)):
        if np.allclose(np.abs(H @ rho_eigvecs[:, i]), np.zeros((H.shape[0], ), dtype=np.float64), atol=tol):
            zero_energy_Dos += local_DoS(rho_eigvecs[:, i], Nsites)
    return zero_energy_Dos
