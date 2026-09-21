"""
Scan the optimal "spin" angle and the marker against the Rashba coupling, and write the
result to data/ for plotting/plot_opt_angle_vs_coupling.py to draw.

    python base-code/03_opt_angle_vs_coupling_v2.py

This script only computes and stores. The figure it used to draw lives in the plotting
script, which reads the .h5 written here, so that the scan does not have to be re-run every
time something about the figure changes.
"""

# Maths
import numpy as np
from numpy import pi

# Saving data
from pathlib import Path
from datetime import date
import sys

# Logging
import logging

# Data
import h5py

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.SO_rashba_dressel_Ham import SO_syst_Kwant
from modules.OPDM import spectrum
from modules.marker import marker_per_site_speedup
from modules.S_optimisation import get_S_tilde_and_gap, S_tilde_linear_basis, S_tilde_gap
from modules.logging_config import setup_logging

#%%
# ============================================================
# Logging setup
# ============================================================

setup_logging()
loger_main = logging.getLogger(__name__)

#%%
# ============================================================
# Code parameters
# ============================================================

M                 = -2.
A                 = 1.
width             = 0.1
W                 = 0.25
r                 = 1.3
Nx                = 30
Ny                = 30
dim_Hext = Nx * Ny
seed     = 12345

# Rashba coupling scan
lambR_vec = np.linspace(0., 1.5, 10)
lambD_vec = [0., 0.5, 0.75, 1.]

# Grid point used for the theta-domain panel (closest to lambR=1, already in lambR_vec)
id_R_fixed = int(np.argmin(np.abs(lambR_vec - 1.)))

# Resolution of the numerical theta optimisation
Ntheta = 50
theta_vec = np.linspace(0, pi / 4, Ntheta)

#%%
# ============================================================
# Main: lattice (PBC), fixed for every (lambR, lambD) point
# ============================================================

loger_main.info('Generating site structure: ...')
lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.seed = seed
lattice.boundary = 'Closed'
lattice.build_lattice()
lattice.generate_onsite_disorder(K_onsite=0.5 * W)
loger_main.info('Generating site structure: Done')

#%%
# ============================================================
# Main: numerically optimal angle and marker for each (lambR, lambD)
# ============================================================

theta_opt_vec            = np.zeros((len(lambD_vec), len(lambR_vec)))
marker_opt_vec           = np.zeros((len(lambD_vec), len(lambR_vec)))
gap_Stilde_opt_vec       = np.zeros((len(lambD_vec), len(lambR_vec)))
gap_H_vec                = np.zeros((len(lambD_vec), len(lambR_vec)))
marker_opt_clean_vec     = np.zeros((len(lambD_vec), len(lambR_vec)))
gap_Stilde_opt_clean_vec = np.zeros((len(lambD_vec), len(lambR_vec)))
theta_opt_clean_vec      = np.zeros((len(lambD_vec), len(lambR_vec)))
gap_theta_fixedR         = np.zeros((len(lambD_vec), Ntheta))

for id_D, lambD in enumerate(lambD_vec):
    for id_R, lambR in enumerate(lambR_vec):
        loger_main.info(f'lambD={lambD}, lambR={lambR:.3f} '
                         f'({id_D * len(lambR_vec) + id_R + 1}/{len(lambD_vec) * len(lambR_vec)})')

        # Model
        params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR, 'lambD': lambD}
        model = SO_syst_Kwant(lattice, params_dict).finalized()
        H = model.hamiltonian_submatrix()
        eps, _, rho = spectrum(H)

        # Spectral gap H
        Nsp = H.shape[0] // 2
        gap_H_vec[id_D, id_R] = eps[Nsp] - eps[Nsp - 1]

        # Optimal numerical theta
        S_tilde_0, S_tilde_1 = S_tilde_linear_basis(rho, dim_Hext)
        gap_theta = np.zeros(theta_vec.shape)
        for i, theta in enumerate(theta_vec):
            gap_theta[i] = S_tilde_gap(S_tilde_0, S_tilde_1, theta)
        theta_opt_vec[id_D, id_R] = theta_vec[np.argmax(gap_theta)]
        if id_R == id_R_fixed:
            gap_theta_fixedR[id_D] = gap_theta

        #  Optimal analytical theta (clean lattice)
        param1 = (lambR + lambD) / A
        param2 = min(0., lambR - lambD) / A
        theta_opt_clean_vec[id_D, id_R] = 0.5 * (np.arctan(param1) + np.arctan(param2))

        # Marker and gap of S_tilde at the optimal angle
        S_tilde_opt, gap_Stilde_opt_vec[id_D, id_R], _ = get_S_tilde_and_gap(
            rho, theta_opt_vec[id_D, id_R], dim_Hext)
        marker_sites = marker_per_site_speedup(lattice.x, lattice.y, S_tilde_opt,
                                               Nx=Nx, Ny=Ny, boundary=lattice.boundary)
        marker_opt_vec[id_D, id_R] = marker_sites.mean()

        # Marker and gap of S_tilde at the clean analytical optimal angle
        S_tilde_clean, gap_Stilde_opt_clean_vec[id_D, id_R], _ = get_S_tilde_and_gap(
            rho, theta_opt_clean_vec[id_D, id_R], dim_Hext)
        marker_sites_clean = marker_per_site_speedup(lattice.x, lattice.y, S_tilde_clean,
                                                     Nx=Nx, Ny=Ny, boundary=lattice.boundary)
        marker_opt_clean_vec[id_D, id_R] = marker_sites_clean.mean()

#%%
# ============================================================
# Saving data
# ============================================================

# Alongside the arrays, everything the figure needs to label itself: the two coupling grids,
# the theta grid, and the index of the lambR point whose theta-domain curve was kept. The
# .h5 is named after the parameters and the date, and never overwritten, so a scan that took
# a while to run cannot be clobbered by the next one
data_dir = Path(__file__).resolve().parent.parent / 'data'
data_dir.mkdir(exist_ok=True)
file_stem = (f'opt_angle_vs_coupling_v2_Nx{Nx}_Ny{Ny}_M{M}_A{A}_W{W}_w{width}'
             f'_{date.today().isoformat()}')
file_path = data_dir / f'{file_stem}.h5'
suffix = 1
while file_path.exists():
    file_path = data_dir / f'{file_stem}_{suffix}.h5'
    suffix += 1

loger_main.info('Saving data...')

with h5py.File(file_path, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'lambR_vec',                lambR_vec)
    store_my_data(simulation, 'lambD_vec',                np.array(lambD_vec))
    store_my_data(simulation, 'theta_vec',                theta_vec)
    store_my_data(simulation, 'theta_opt_vec',            theta_opt_vec)
    store_my_data(simulation, 'theta_opt_clean_vec',      theta_opt_clean_vec)
    store_my_data(simulation, 'marker_opt_vec',           marker_opt_vec)
    store_my_data(simulation, 'marker_opt_clean_vec',     marker_opt_clean_vec)
    store_my_data(simulation, 'gap_Stilde_opt_vec',       gap_Stilde_opt_vec)
    store_my_data(simulation, 'gap_Stilde_opt_clean_vec', gap_Stilde_opt_clean_vec)
    store_my_data(simulation, 'gap_H_vec',                gap_H_vec)
    store_my_data(simulation, 'gap_theta_fixedR',         gap_theta_fixedR)
    store_my_data(simulation, 'id_R_fixed',               id_R_fixed)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'M',      M)
    store_my_data(parameters, 'A',      A)
    store_my_data(parameters, 'width',  width)
    store_my_data(parameters, 'W',      W)
    store_my_data(parameters, 'r',      r)
    store_my_data(parameters, 'Nx',     Nx)
    store_my_data(parameters, 'Ny',     Ny)
    store_my_data(parameters, 'seed',   seed)
    store_my_data(parameters, 'Ntheta', Ntheta)

    # Attributes
    attr_my_data(parameters, "Date",      str(date.today()))
    attr_my_data(parameters, "Code_path", sys.argv[0])

loger_main.info(f'Data saved correctly to {file_path}')
