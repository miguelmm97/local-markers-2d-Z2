import numpy as np

from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.SO_rashba_dressel_Ham import SO_syst_Kwant
from modules.OPDM import spectrum
from modules.S_optimisation import S_tilde_one_parameter


def _small_rho(Nx=4, Ny=4, lambR=1., lambD=0.):
    lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=0., r=1.3)
    lattice.build_lattice()
    params_dict = {'M': -2., 'W': 0., 'A': 1., 'lambR': lambR, 'lambD': lambD}
    model = SO_syst_Kwant(lattice, params_dict).finalized()
    H = model.hamiltonian_submatrix()
    _, _, rho = spectrum(H)
    return rho, Nx * Ny


def test_flattened_S_tilde_is_involutive():
    # The whole point of the sign-flattening step is S_tilde ** 2 = I. If this
    # ever stops holding, the marker built on top of it is meaningless
    rho, dim_Hext = _small_rho()
    S_tilde, gap, vals = S_tilde_one_parameter(rho, theta=0.4, dim_Hext=dim_Hext)
    D = S_tilde.shape[0]
    assert np.allclose(S_tilde @ S_tilde, np.eye(D), atol=1e-8)


def test_flattened_S_tilde_is_hermitian():
    rho, dim_Hext = _small_rho()
    S_tilde, gap, vals = S_tilde_one_parameter(rho, theta=0.4, dim_Hext=dim_Hext)
    assert np.allclose(S_tilde, S_tilde.conj().T, atol=1e-10)


def test_gap_matches_raw_spectrum():
    rho, dim_Hext = _small_rho()
    _, gap, vals = S_tilde_one_parameter(rho, theta=0.4, dim_Hext=dim_Hext)
    assert gap == np.min(np.abs(vals))
    assert gap >= 0.
