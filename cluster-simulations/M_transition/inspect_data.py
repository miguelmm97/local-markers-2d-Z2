"""
Sanity-check a batch of finished jobs.

    python inspect_data.py data-10            # summarise one size
    python inspect_data.py data-*             # summarise several
    python inspect_data.py data-10 --full     # also dump the structure of the first file

Checks the things that actually go wrong in an array job: tasks that never wrote a file
(killed by walltime or the memory cap), NaNs, and samples that came out identical to each
other -- which would mean the disorder realisations are not being varied. It also reports
which candidate angles won, since a scan that always picks the same one is a sign that the
optimisation is not buying anything.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

parser = argparse.ArgumentParser(description='Sanity-check finished M transition jobs')
parser.add_argument('dirs', nargs='+', help='Data directories to inspect')
parser.add_argument('--full', action='store_true', help='Dump the structure of the first file')
args = parser.parse_args()


def describe(path, indent='    '):
    """Print the tree of groups, datasets and attributes in one file."""
    with h5py.File(path, 'r') as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                value = obj[()]
                if obj.shape == ():
                    shown = value.decode() if isinstance(value, bytes) else value
                else:
                    shown = f'shape={obj.shape} dtype={obj.dtype}'
                print(f'{indent}{name:<28} {shown}')
        f.visititems(walk)
        for group in f:
            for key, value in f[group].attrs.items():
                print(f'{indent}{group}.attrs[{key}] = {value}')


exit_code = 0
for directory in args.dirs:
    directory = Path(directory)
    files = sorted(directory.glob('*.h5'), key=lambda p: int(p.stem.split('-')[-1]))

    print(f'\n{directory}  ({len(files)} files)')
    if not files:
        print('    no .h5 files found')
        exit_code = 1
        continue

    markers, thetas, samples, sizes, lines = [], [], [], [], []
    theta_k = None
    for path in files:
        try:
            with h5py.File(path, 'r') as f:
                markers.append(f['Simulation/marker'][:])
                thetas.append(f['Simulation/theta_opt'][:])
                samples.append(int(f['Parameters/sample'][()]))
                sizes.append(int(f['Parameters/N'][()]))
                theta_k = float(f['Parameters/theta_k'][()])
                lines.append(int(path.stem.split('-')[-1]))
        except Exception as exc:
            print(f'    UNREADABLE {path.name}: {exc}')
            exit_code = 1

    if not markers:
        continue
    markers = np.array(markers)
    thetas = np.array(thetas)

    # Array tasks that never wrote a file leave a gap in the line numbering
    expected = set(range(min(lines), max(lines) + 1))
    missing = sorted(expected - set(lines))
    if missing:
        print(f'    MISSING lines (failed or still running): {missing}')
        exit_code = 1

    if len(set(sizes)) != 1:
        print(f'    MIXED system sizes in one directory: {sorted(set(sizes))}')
        exit_code = 1

    n_bad = int(np.sum(~np.isfinite(markers)))
    if n_bad:
        print(f'    {n_bad} non-finite marker values')
        exit_code = 1

    # If the per-sample seed is not taking effect, every realisation is bit-identical
    if len(markers) > 1:
        spread = float(np.abs(markers - markers[0]).max())
        if spread == 0.0:
            print('    ALL SAMPLES IDENTICAL -- the disorder seed is not varying')
            exit_code = 1

    mean = markers.mean(axis=0)
    std = markers.std(axis=0)
    print(f'    L={sizes[0]}, samples {min(samples)}-{max(samples)}, '
          f'{markers.shape[1]} M points')
    print(f'    marker: starts {mean[0]:+.3f} +/- {std[0]:.3f}, '
          f'ends {mean[-1]:+.3f} +/- {std[-1]:.3f}')
    print(f'    range over all samples: [{markers.min():+.3f}, {markers.max():+.3f}], '
          f'largest spread between samples {np.abs(markers - markers[0]).max():.3f}')

    # Which angles actually won
    unique, counts = np.unique(np.round(thetas, 6), return_counts=True)
    share = ', '.join(f'{u:.4f}: {100 * c / thetas.size:.0f}%' for u, c in zip(unique, counts))
    print(f'    theta_k={theta_k:.4f}; chosen angles -> {share}')
    if len(unique) == 1:
        print('    (only one angle ever won -- the theta scan is not changing the result)')

    if args.full:
        print(f'\n    structure of {files[0].name}:')
        describe(files[0])

sys.exit(exit_code)
