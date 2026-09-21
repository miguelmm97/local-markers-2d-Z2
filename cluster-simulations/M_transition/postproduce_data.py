"""
Collapse a finished batch into a single, small .h5 file that can be copied off the cluster
and read straight by a plotting script.

    python postproduce_data.py runs/2026-09-21_1200
    python postproduce_data.py runs/2026-09-21_1200 -o ~/M_transition.h5
    python postproduce_data.py runs/2026-09-21_1200 --quantiles 0.1 0.25 0.75 0.9

The raw batch is one .h5 per (system size, disorder sample), each holding the marker and the
two gaps along the same M grid. For every size this script stacks its samples and reduces
them along the sample axis -- mean, std, standard error, median, min, max and the requested
quantiles -- so what comes out is a handful of (n_sizes, NM) arrays ready to be plotted.

Three other things travel with the statistics so that the plotting script never has to look
at the run directory again:

  * the parameters of the batch, taken from the .toml snapshot written into the run
    directory by generate_submits.py (falling back to config/ if the snapshot is missing),
    including the raw text of the file;
  * the per-sample curves, seeds and job line numbers, kept unreduced. They are tiny
    (Nsamples x NM floats per size) and they are the only way to go back from an outlier in
    an averaged curve to the realisation that produced it;
  * which array tasks never wrote a file, so a partially finished batch is obvious at
    plotting time rather than silently averaged over fewer samples.

The output file holds three groups:

    Simulation/   M, size_vec, quantiles, n_samples, and <obs>_<stat> for every observable
                  ('marker', 'Gamma_gap', 'H_gap', 'theta_opt') and every statistic ('avg',
                  'std', 'sem', 'median', 'min', 'max', 'n_valid', 'quantiles'). All are
                  (n_sizes, NM), except the quantiles, which are (n_sizes, n_quantiles, NM)
    Samples/L<L>/ the unreduced (n_samples, NM) curves plus sample, job_seed, line and
                  missing_lines. Row i of every array is the realisation with seed
                  job_seed[i]
    Parameters/   every key of the .toml, plus theta_k, its raw text, and the sizes actually
                  found in the batch

so a plot is roughly

    with h5py.File('M_transition_summary.h5', 'r') as f:
        M        = f['Simulation/M'][:]
        size_vec = f['Simulation/size_vec'][:]
        avg      = f['Simulation/marker_avg'][:]
        band     = f['Simulation/marker_quantiles'][:]     # (n_sizes, n_quantiles, NM)
        W        = f['Parameters/W'][()]
    for id_L, L in enumerate(size_vec):
        ax.plot(M, avg[id_L], label=f'${L}$')
        ax.fill_between(M, band[id_L, 0], band[id_L, -1], alpha=0.2)

Note on the marker: the lattices are run with closed (periodic) boundaries, so every site is
a bulk site and the per-file 'marker' is already the average over the whole lattice -- there
is no boundary to cut away here. The distribution reduced below is therefore the
distribution over disorder realisations at each M, not over sites.
"""

# Utilities
import argparse
import sys
from datetime import date
from pathlib import Path

# The repo root (where 'modules' lives) is two levels up from this file, same as in the
# cluster script, so this runs from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Maths
import numpy as np

# Logging
import logging

# Data
import h5py
import tomli

# Modules
from modules.functions import store_my_data, attr_my_data
from modules.logging_config import setup_logging


parser = argparse.ArgumentParser(description='Reduce a batch of M transition jobs to one .h5 file')
parser.add_argument('run_dir', type=str, help='Run directory (or any directory holding data-<L> folders)')
parser.add_argument('-o', '--out', type=str, default=None,
                    help='Output file (default: <run_dir>/M_transition_summary.h5)')
parser.add_argument('-q', '--quantiles', type=float, nargs='+', default=[0.05, 0.25, 0.75, 0.95],
                    help='Quantiles of the sample distribution to store (default: 0.05 0.25 0.75 0.95)')
args = parser.parse_args()


#%%
# ============================================================
# Logging setup
# ============================================================

setup_logging()
loger_main = logging.getLogger(__name__)


#%%
# ============================================================
# Locating the batch
# ============================================================

# Observables reduced along the sample axis. theta_opt is not an observable in the physical
# sense, but its spread tells at a glance whether the angle scan is doing anything
OBSERVABLES = ['marker', 'Gamma_gap', 'H_gap', 'theta_opt']

run_dir = Path(args.run_dir).expanduser().resolve()
if not run_dir.is_dir():
    raise IOError(f'{run_dir} is not a directory')

# generate_submits.py puts the data under <run_dir>/data/data-<L>, but older batches kept
# data-<L> at the top level, and pointing the script at the data folder itself should work too
data_dirs = sorted(run_dir.glob('data/data-*')) or sorted(run_dir.glob('data-*'))
if not data_dirs:
    raise IOError(f'No data-<L> directories found under {run_dir}')
data_dirs = sorted((d for d in data_dirs if d.is_dir()), key=lambda d: int(d.name.split('-')[-1]))
loger_main.info(f'Found {len(data_dirs)} system sizes under {run_dir}')

quantiles = np.sort(np.asarray(args.quantiles, dtype=float))
if np.any(quantiles < 0.) or np.any(quantiles > 1.):
    raise ValueError(f'Quantiles must lie in [0, 1], got {args.quantiles}')

out_path = Path(args.out).expanduser() if args.out else run_dir / 'M_transition_summary.h5'
out_path.parent.mkdir(parents=True, exist_ok=True)


#%%
# ============================================================
# Parameters of the batch
# ============================================================

# The snapshot inside the run directory is what the batch actually ran with; config/ may
# have been edited since. Fall back to it only if the snapshot is not there
snapshot = run_dir / 'parameters_M_transition.toml'
if not snapshot.exists():
    snapshot = Path(__file__).resolve().parent / 'config' / 'parameters_M_transition.toml'
    loger_main.warning(f'No config snapshot in the run directory, falling back to {snapshot}')
loger_main.info(f'Reading parameters from {snapshot}')

with snapshot.open('rb') as fp:
    params = tomli.load(fp)
params_text = snapshot.read_text()

# jobs.txt maps array task id -> (size, sample), which is what says whether a file is
# missing rather than simply never having been asked for. Without it, fall back to the
# size-major block layout that generate_jobs.py writes from the same config
jobs_path = run_dir / 'jobs.txt'
expected_lines = {}
if jobs_path.exists():
    for line, entry in enumerate(jobs_path.read_text().splitlines()):
        size, sample = entry.split()
        expected_lines.setdefault(int(size), []).append(line)
else:
    loger_main.warning(f'No jobs.txt in {run_dir}, assuming the size-major layout of the config')
    for id_L, size in enumerate(params['size_vec']):
        expected_lines[size] = list(range(id_L * params['Nsamples'],
                                          (id_L + 1) * params['Nsamples']))


#%%
# ============================================================
# Reading the batch
# ============================================================

def read_size(directory):
    """Stack every sample of one system size into (n_samples, NM) arrays.

    Files are read in order of the disorder sample rather than of the file name, so that a
    row index means the same realisation in every array returned, and so that seeds[i]
    identifies the realisation behind row i of each observable.

    Returns None if the size has nothing readable yet, so that a batch can be reduced while
    it is still running. A file that cannot be opened -- a job killed halfway through its
    write, or one being written at this very moment -- is skipped rather than taken down
    with the whole script, and reappears in missing_lines like a job that never ran.
    """

    files = sorted(directory.glob('*.h5'), key=lambda p: int(p.stem.split('-')[-1]))
    if not files:
        loger_main.warning(f'    no .h5 files in {directory} yet -- skipping this size')
        return None

    records = []
    for path in files:
        try:
            with h5py.File(path, 'r') as f:
                record = {obs: f[f'Simulation/{obs}'][:] for obs in OBSERVABLES}
                record['M'] = f['Simulation/M'][:]
                record['N'] = int(f['Parameters/N'][()])
                record['sample'] = int(f['Parameters/sample'][()])
                record['job_seed'] = int(f['Parameters/job_seed'][()])
                record['theta_k'] = float(f['Parameters/theta_k'][()])
                record['line'] = int(path.stem.split('-')[-1])
        except (OSError, KeyError) as exc:
            loger_main.warning(f'    UNREADABLE {path.name} ({exc}) -- skipping it')
            continue
        records.append(record)

    if not records:
        loger_main.warning(f'    nothing readable in {directory} -- skipping this size')
        return None

    records.sort(key=lambda r: r['sample'])

    sizes = {r['N'] for r in records}
    if len(sizes) != 1:
        raise ValueError(f'{directory} mixes system sizes {sorted(sizes)}')

    # Every file of the batch walks the same M grid, since it is built from the config, and
    # the stacking below is only meaningful if that holds
    M = records[0]['M']
    for r in records[1:]:
        if r['M'].shape != M.shape or not np.allclose(r['M'], M):
            raise ValueError(f'{directory}: line {r["line"]} was run on a different M grid')

    data = {obs: np.array([r[obs] for r in records], dtype=float) for obs in OBSERVABLES}
    data['M'] = M
    data['N'] = records[0]['N']
    data['theta_k'] = records[0]['theta_k']
    data['sample'] = np.array([r['sample'] for r in records], dtype=int)
    data['job_seed'] = np.array([r['job_seed'] for r in records], dtype=int)
    data['line'] = np.array([r['line'] for r in records], dtype=int)
    return data


def statistics(data, quantiles):
    """Reduce (n_samples, NM) along the sample axis.

    Non-finite entries are turned into NaN and skipped, so that one blown-up realisation
    costs that M point one sample instead of poisoning the whole curve. The spread is the
    unbiased (ddof=1) estimate, since these are disorder realisations drawn from a
    distribution rather than the whole of it, and 'sem' is the error of the mean that an
    error bar around 'avg' should use. 'n_valid' says how many samples each point was
    actually averaged over, which is where a discarded realisation shows up.
    """

    clean = np.where(np.isfinite(data), data, np.nan)
    n_valid = np.sum(np.isfinite(clean), axis=0)
    ddof = 1 if clean.shape[0] > 1 else 0

    with np.errstate(invalid='ignore'):
        std = np.nanstd(clean, axis=0, ddof=ddof)
        stats = {'avg':       np.nanmean(clean, axis=0),
                 'n_valid':   n_valid,
                 'std':       std,
                 'sem':       np.divide(std, np.sqrt(n_valid), out=np.full_like(std, np.nan),
                                        where=n_valid > 0),
                 'median':    np.nanmedian(clean, axis=0),
                 'min':       np.nanmin(clean, axis=0),
                 'max':       np.nanmax(clean, axis=0),
                 'quantiles': np.nanquantile(clean, quantiles, axis=0)}
    return stats


size_vec, batch, missing = [], {}, {}
for directory in data_dirs:
    loger_main.info(f'Reading {directory} ...')
    data = read_size(directory)
    if data is None:
        continue
    L = data['N']

    if L in batch:
        raise ValueError(f'System size L={L} appears in more than one data directory')
    size_vec.append(L)
    batch[L] = data

    # Array tasks that died leave a gap in the line numbering. Reporting them here is what
    # keeps a half-finished batch from being averaged as if it were complete
    missing[L] = sorted(set(expected_lines.get(L, [])) - set(data['line'].tolist()))
    n_bad = int(np.sum(~np.isfinite(data['marker'])))
    loger_main.info(f'    L={L}: {len(data["sample"])} samples, {len(data["M"])} M points'
                    + (f', {len(missing[L])} MISSING lines {missing[L]}' if missing[L] else '')
                    + (f', {n_bad} non-finite marker values' if n_bad else ''))

if not size_vec:
    raise IOError(f'No readable .h5 files anywhere under {run_dir} -- has the batch started?')

# Sizes that finished a different number of samples are fine: each one is reduced on its own
# and only then stacked, so an unfinished size simply averages over the samples it has.
# Simulation/n_samples is what says how many that was
if len({len(batch[L]['sample']) for L in size_vec}) != 1:
    loger_main.warning('The sizes did not all finish the same number of samples '
                       '-- see Simulation/n_samples and Samples/L*/missing_lines')

# The sizes share the M grid too, which is what lets the statistics be stacked into one
# (n_sizes, NM) array per observable instead of one group per size
M_vec = batch[size_vec[0]]['M']
for L in size_vec[1:]:
    if batch[L]['M'].shape != M_vec.shape or not np.allclose(batch[L]['M'], M_vec):
        raise ValueError(f'L={L} was run on a different M grid than L={size_vec[0]}')

theta_k = batch[size_vec[0]]['theta_k']


#%%
# ============================================================
# Statistics
# ============================================================

loger_main.info('Calculating statistics...')

# One array per (observable, statistic), stacked over sizes so that a plotting script can
# index it the same way the single-machine version of this scan does: array[id_L]
stats = {obs: {} for obs in OBSERVABLES}
for obs in OBSERVABLES:
    per_size = [statistics(batch[L][obs], quantiles) for L in size_vec]
    for key in per_size[0]:
        # 'quantiles' is (n_quantiles, NM) per size, so stacking puts the size axis first
        # here as well: (n_sizes, n_quantiles, NM)
        stats[obs][key] = np.stack([s[key] for s in per_size], axis=0)

for L in size_vec:
    avg = np.nanmean(np.where(np.isfinite(batch[L]['marker']), batch[L]['marker'], np.nan), axis=0)
    loger_main.info(f'    L={L}: marker goes from {avg[0]:+.3f} at M={M_vec[0]:.2f} '
                    f'to {avg[-1]:+.3f} at M={M_vec[-1]:.2f}')


#%%
# ============================================================
# Saving data
# ============================================================

loger_main.info(f'Saving data to {out_path} ...')

with h5py.File(out_path, 'w') as f:

    # Statistics folder: everything a plot needs, stacked over system sizes
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'M',         M_vec)
    store_my_data(simulation, 'size_vec',  np.array(size_vec, dtype=int))
    store_my_data(simulation, 'quantiles', quantiles)
    store_my_data(simulation, 'n_samples', np.array([len(batch[L]['sample']) for L in size_vec], dtype=int))
    for obs in OBSERVABLES:
        for key, value in stats[obs].items():
            store_my_data(simulation, f'{obs}_{key}', value)

    attr_my_data(simulation, 'Observables',   OBSERVABLES)
    attr_my_data(simulation, 'Reduced_axis',  'disorder samples, at fixed (L, M)')
    attr_my_data(simulation, 'Array_layout',  '(n_sizes, NM), and (n_sizes, n_quantiles, NM) for *_quantiles')
    attr_my_data(simulation, 'std_ddof',      1)

    # Per-sample folder: the unreduced curves with the seed that produced each of them, so
    # an outlier in the averages can be traced back to its realisation and re-run
    samples = f.create_group('Samples')
    for L in size_vec:
        group = samples.create_group(f'L{L}')
        store_my_data(group, 'sample',   batch[L]['sample'])
        store_my_data(group, 'job_seed', batch[L]['job_seed'])
        store_my_data(group, 'line',     batch[L]['line'])
        store_my_data(group, 'missing_lines', np.array(missing[L], dtype=int))
        for obs in OBSERVABLES:
            store_my_data(group, obs, batch[L][obs])
        attr_my_data(group, 'Row_meaning', 'row i of every array is disorder sample sample[i], seed job_seed[i]')

    # Parameters folder: the config the batch ran with, so the plotting script never needs
    # the run directory. The raw text goes in as well, since a parameter added to the .toml
    # later would otherwise be lost on files written by this version of the script
    parameters = f.create_group('Parameters')
    for key, value in params.items():
        store_my_data(parameters, key, int(value) if isinstance(value, bool) else value)
    store_my_data(parameters, 'theta_k',      theta_k)
    store_my_data(parameters, 'toml',         params_text)
    store_my_data(parameters, 'size_vec_data', np.array(size_vec, dtype=int))

    attr_my_data(parameters, 'Date',       str(date.today()))
    attr_my_data(parameters, 'Code_path',  sys.argv[0])
    attr_my_data(parameters, 'Run_dir',    str(run_dir))
    attr_my_data(parameters, 'Config',     str(snapshot))
    attr_my_data(parameters, 'Note',       'size_vec is what the config asked for, '
                                           'size_vec_data is what was actually found in the batch')

total_missing = sum(len(v) for v in missing.values())
if total_missing:
    loger_main.warning(f'{total_missing} jobs never wrote a file -- the averages are over '
                       f'fewer samples than the config asked for (see Samples/L*/missing_lines)')
loger_main.info(f'Data saved correctly to {out_path}')
