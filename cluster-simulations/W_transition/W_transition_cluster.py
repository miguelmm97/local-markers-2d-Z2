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
from modules.S_optimisation import get_S_tilde_and_gap
from modules.logging_config import setup_logging

# Data
import h5py


# Arguments to submit to the cluster
parser = argparse.ArgumentParser(description='Local marker vs onsite disorder strength')
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
params = config.parameters_W_transition

M        = params['M']
A        = params['A']
width    = params['width']
r        = params['r']
seed     = params['seed']
lambR    = params['lambR']
lambD    = params['lambD']
W0       = params['W0']
Wf       = params['Wf']
NW       = params['NW']
W_vec = np.linspace(W0, Wf, NW)

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

# Seed for this particular disorder realisation. AmorphousLattice_2d draws both the amorphous
# site positions and the Anderson onsite disorder from self.seed, so a sample-dependent seed
# averages over both. Keeping the base seed from the .toml in the offset makes the whole batch
# reproducible, and the stride is larger than any realistic Nsamples so that the seeds used by
# different system sizes never collide
job_seed = seed + 1000003 * sample

# Analytical (clean-lattice) angle at this (lambR, lambD)
param1 = (lambR + lambD) / A
param2 = min(0., lambR - lambD) / A
theta_k = 0.5 * (np.arctan(param1) + np.arctan(param2))

#%%
# ============================================================
# Main: marker and gap of S_tilde at theta_k, for each disorder strength W
# ============================================================

marker_vec = np.zeros(len(W_vec))
gap_vec    = np.zeros(len(W_vec))
gap_H_vec  = np.zeros(len(W_vec))

# Amorphous lattice
Nx = Ny = N
dim_Hext = Nx * Ny
loger_main.info(f'Generating site structure for L={N}, sample={sample} (seed={job_seed}): ...')
lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
lattice.seed = job_seed
lattice.boundary = 'Closed'
lattice.build_lattice()
loger_main.info(f'Generating site structure for L={N}, sample={sample}: Done')


for i, W in enumerate(W_vec):
    loger_main.info(f'L={N}, sample={sample}, W={W:.3f} ({i + 1}/{len(W_vec)})')

    # Same seed every call -> same underlying random pattern, just rescaled by K_onsite,
    # so the W scan follows one disorder realisation instead of jumping between them
    lattice.generate_onsite_disorder(K_onsite=0.5 * W)
    params_dict = {'M': M, 'W': W, 'A': A, 'lambR': lambR, 'lambD': lambD}
    model = SO_syst_Kwant(lattice, params_dict).finalized()
    H = model.hamiltonian_submatrix()

    # The eigenvectors of H are not needed once the projector is built, and at these system
    # sizes they are a multi-GB array, so they are dropped before S_tilde allocates its own
    energy, eigenstates, rho = spectrum(H)
    del eigenstates, H

    # Gap of the Hamiltonian spectrum around the Fermi level (half filling)
    Nsp = int(len(rho) / 2)
    gap_H_vec[i] = energy[Nsp] - energy[Nsp - 1]

    # Marker and gap of S_tilde at the fixed analytical angle
    S_tilde, gap_vec[i], _ = get_S_tilde_and_gap(rho, theta_k, dim_Hext)
    del rho
    marker_sites = marker_per_site_speedup(lattice.x, lattice.y, S_tilde, Nx=Nx, Ny=Ny,
                                           boundary=lattice.boundary)
    marker_vec[i] = marker_sites.mean()
    del S_tilde


#%% Saving data

os.makedirs(args.outdir, exist_ok=True)
outfile = '{}-{}.h5'.format(args.outbase, args.line)
filepath = os.path.join(args.outdir, outfile)

loger_main.info('Saving data...')

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'marker',    marker_vec)
    store_my_data(simulation, 'H_gap',     gap_H_vec)
    store_my_data(simulation, 'Gamma_gap', gap_vec)
    store_my_data(simulation, 'W',         W_vec)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters,    'N',         N)
    store_my_data(parameters,    'sample',    sample)
    store_my_data(parameters,    'job_seed',  job_seed)
    store_my_data(parameters,    'M',         M)
    store_my_data(parameters,    'A',         A)
    store_my_data(parameters,    'width',     width)
    store_my_data(parameters,    'r',         r)
    store_my_data(parameters,    'seed',      seed)
    store_my_data(parameters,    'lambR',     lambR)
    store_my_data(parameters,    'lambD',     lambD)
    store_my_data(parameters,    'W0',        W0)
    store_my_data(parameters,    'Wf',        Wf)
    store_my_data(parameters,    'NW',        NW)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info(f'Data saved correctly to {filepath}')
