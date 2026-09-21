# Utilities
import argparse
import os
import sys
from datetime import date
from pathlib import Path

# The repo root (where 'modules' lives) is two levels up from this file. Inserting it
# explicitly means the job runs correctly whatever the working directory Slurm gives it,
# without relying on PYTHONPATH being set in the submit script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Maths
import numpy as np

# Logging
import logging

# Modules
import config
from modules.functions import store_my_data, attr_my_data
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.SO_rashba_dressel_Ham import SO_syst_Kwant
from modules.OPDM import spectrum
from modules.marker import marker_per_site_speedup
from modules.S_optimisation import get_S_tilde_and_gap, S_tilde_linear_basis, S_tilde_gap
from modules.logging_config import setup_logging

# Data
import h5py


# Arguments to submit to the cluster
parser = argparse.ArgumentParser(description='Local marker vs onsite mass')
parser.add_argument('-l', '--line', type=int, help='Select line number', default=None)
parser.add_argument('-f', '--file', type=str, help='Select file name', default='jobs.txt')
parser.add_argument('-M', '--outdir', type=str, help='Select the output directory', default='outdir')
parser.add_argument('-o', '--outbase', type=str, help='Select the base name of the output file', default='Job')
args = parser.parse_args()


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

# Parameters common to the whole simulation batch
loger_main.info('Loading variables from .toml file')
params = config.parameters_M_transition

A              = params['A']
width          = params['width']
r              = params['r']
seed           = params['seed']
lambR          = params['lambR']
lambD          = params['lambD']
W              = params['W']
M0             = params['M0']
Mf             = params['Mf']
NM             = params['NM']
Mc0            = params['Mc0']
Mc1            = params['Mc1']
dense_fraction = params['dense_fraction']
optimize_theta = params['optimize_theta']
Ntheta         = params['Ntheta']


def concentrated_grid(x0, xf, xc0, xc1, N, dense_fraction):
    # N points on [x0, xf], of which dense_fraction sit inside the window [xc0, xc1]. The
    # rest are shared between the two outer stretches in proportion to their width, so
    # neither of them ends up coarser than the other. The joins are duplicated by
    # construction and removed by unique, which also returns the grid sorted
    n_dense = int(round(dense_fraction * N))
    w_low, w_high = xc0 - x0, xf - xc1
    n_outer = N - n_dense
    n_low = int(round(n_outer * w_low / (w_low + w_high)))
    n_high = n_outer - n_low
    return np.unique(np.concatenate([np.linspace(x0, xc0, n_low + 1),
                                     np.linspace(xc0, xc1, n_dense),
                                     np.linspace(xc1, xf, n_high + 1)]))
M_vec = concentrated_grid(M0, Mf, Mc0, Mc1, NM, dense_fraction)
loger_main.info(f'M grid: {len(M_vec)} points, {int(round(dense_fraction * NM))} of them '
                f'inside [{Mc0}, {Mc1}]')

# Specific parameters for this job
loger_main.info('Loading variables from parameter file')
if args.line is None:
    raise IOError('No line number was given')

with open(args.file, 'r') as f:
    job_lines = f.read().splitlines()

if not 0 <= args.line < len(job_lines):
    raise IndexError(f'Line {args.line} is out of range for {args.file}, '
                     f'which has {len(job_lines)} lines')

job_params = job_lines[args.line].split()
if len(job_params) != 2:
    raise ValueError(f'Line {args.line} of {args.file} should hold "<size> <sample>", '
                     f'got {job_lines[args.line]!r}')
N, sample = int(job_params[0]), int(job_params[1])
loger_main.info(f'Line number: {args.line} -> L={N}, sample={sample}')

# Seed for this particular disorder realisation.
job_seed = seed + 1000003 * sample

# Analytical (clean-lattice) angle at this (lambR, lambD).
param1 = (lambR + lambD) / A
param2 = min(0., lambR - lambD) / A
theta_k = 0.5 * (np.arctan(param1) + np.arctan(param2))

# Candidate angles.
if optimize_theta:
    if Ntheta < 2:
        raise ValueError(f'Ntheta must be at least 2 to span [0, theta_k], got {Ntheta}')
    theta_candidates = np.linspace(0., theta_k, Ntheta)
else:
    theta_candidates = np.array([theta_k])
loger_main.info(f'theta_k={theta_k:.4f}, optimize_theta={optimize_theta}, '
                f'candidates={np.array2string(theta_candidates, precision=4)}')

#%%
# ============================================================
# Main: marker and widest gap of S_tilde over the candidate angles, for each mass M
# ============================================================

marker_vec      = np.zeros(len(M_vec))
gap_vec         = np.zeros(len(M_vec))
gap_H_vec       = np.zeros(len(M_vec))
theta_opt_vec   = np.zeros(len(M_vec))
marker_sites    = np.zeros((int(N ** 2), len(M_vec)))

# Amorphous lattice. The disorder strength is fixed for the whole scan
Nx = Ny = N
dim_Hext = Nx * Ny
loger_main.info(f'Generating site structure for L={N}, sample={sample} (seed={job_seed}): ...')
lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.seed = job_seed
lattice.boundary = 'Closed'
lattice.build_lattice()
lattice.generate_onsite_disorder(K_onsite=0.5 * W)
loger_main.info(f'Generating site structure for L={N}, sample={sample}: Done')


for i, M in enumerate(M_vec):
    loger_main.info(f'L={N}, sample={sample}, M={M:.3f} ({i + 1}/{len(M_vec)})')

    # Model
    params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR, 'lambD': lambD}
    model = SO_syst_Kwant(lattice, params_dict).finalized()
    H = model.hamiltonian_submatrix()

    # Gap of the Hamiltonian and rho
    energy, eigenstates, rho = spectrum(H)
    del eigenstates, H
    Nsp = int(len(rho) / 2)
    gap_H_vec[i] = energy[Nsp] - energy[Nsp - 1]

    # Pick the candidate angle with the widest gap
    if len(theta_candidates) == 1:
        theta_opt = theta_candidates[0]
    else:
        S_tilde_0, S_tilde_1 = S_tilde_linear_basis(rho, dim_Hext)
        gaps = np.array([S_tilde_gap(S_tilde_0, S_tilde_1, t) for t in theta_candidates])
        del S_tilde_0, S_tilde_1
        theta_opt = theta_candidates[np.argmax(gaps)]
        loger_main.info(f'gap_gamma {np.max(gaps):.3f}')
    theta_opt_vec[i] = theta_opt


    # Marker and gap of S_tilde at the chosen angle
    S_tilde, gap_vec[i], _ = get_S_tilde_and_gap(rho, theta_opt, dim_Hext)
    del rho
    marker_sites[:, i] = marker_per_site_speedup(lattice.x, lattice.y, S_tilde, Nx=Nx, Ny=Ny, boundary=lattice.boundary)
    marker_vec[i] = marker_sites[:, i].mean()
    del S_tilde


#%% Saving data

os.makedirs(args.outdir, exist_ok=True)
outfile = '{}-{}.h5'.format(args.outbase, args.line)
filepath = os.path.join(args.outdir, outfile)

loger_main.info('Saving data...')

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'marker',          marker_vec)
    store_my_data(simulation, 'marker_per_site', marker_sites)
    store_my_data(simulation, 'H_gap',           gap_H_vec)
    store_my_data(simulation, 'Gamma_gap',       gap_vec)
    store_my_data(simulation, 'theta_opt',       theta_opt_vec)
    store_my_data(simulation, 'M',               M_vec)
    store_my_data(simulation, 'x',               lattice.x)
    store_my_data(simulation, 'y',               lattice.y)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters,    'N',              N)
    store_my_data(parameters,    'sample',         sample)
    store_my_data(parameters,    'job_seed',       job_seed)
    store_my_data(parameters,    'A',              A)
    store_my_data(parameters,    'width',          width)
    store_my_data(parameters,    'r',              r)
    store_my_data(parameters,    'seed',           seed)
    store_my_data(parameters,    'lambR',          lambR)
    store_my_data(parameters,    'lambD',          lambD)
    store_my_data(parameters,    'W',              W)
    store_my_data(parameters,    'M0',             M0)
    store_my_data(parameters,    'Mf',             Mf)
    store_my_data(parameters,    'NM',             NM)
    store_my_data(parameters,    'Mc0',            Mc0)
    store_my_data(parameters,    'Mc1',            Mc1)
    store_my_data(parameters,    'dense_fraction', dense_fraction)
    store_my_data(parameters,    'theta_k',        theta_k)
    store_my_data(parameters,    'optimize_theta', int(optimize_theta))
    store_my_data(parameters,    'Ntheta',         len(theta_candidates))

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info(f'Data saved correctly to {filepath}')
