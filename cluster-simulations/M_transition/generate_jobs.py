"""
Write jobs.txt, one line per (system size, disorder sample).

The line number is the Slurm array task id, so the file must stay headerless and the
ordering must stay size-major: generate_submits.py turns each size's block of lines into
one contiguous --array range.
"""

from pathlib import Path

import config

parameters = config.parameters_M_transition
jobs_path = Path(__file__).resolve().parent / 'jobs.txt'

with open(jobs_path, 'w') as f:
    for size in parameters['size_vec']:
        for sample in range(parameters['Nsamples']):
            f.write(f'{size} {sample}\n')

n_jobs = len(parameters['size_vec']) * parameters['Nsamples']
print(f'Wrote {n_jobs} jobs to {jobs_path}')
