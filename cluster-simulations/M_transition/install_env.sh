#!/bin/bash
#
# Create the conda environment needed to run the W transition jobs.
#
#       ./install_env.sh [env_name]
#
# Kwant and tinyarray are compiled packages that are only packaged on conda-forge, so the
# whole environment is resolved from conda-forge to keep the BLAS and the compiled
# extensions consistent with each other.

set -euo pipefail

ENV_NAME="${1:-local-markers}"
PYTHON_VERSION="3.12"

if ! command -v conda &> /dev/null; then
    echo "conda not found. Load your cluster's anaconda/miniforge module first, e.g.:"
    echo "    module load anaconda3"
    exit 1
fi

# conda activate is a shell function, not on PATH in a non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    echo "Remove it first with 'conda env remove -n $ENV_NAME' if you want a clean rebuild."
    exit 1
fi

echo "Creating conda environment '$ENV_NAME' (python $PYTHON_VERSION)..."
conda create -y -n "$ENV_NAME" -c conda-forge \
    python="$PYTHON_VERSION" \
    numpy \
    scipy \
    kwant \
    tinyarray \
    h5py \
    tomli \
    matplotlib \
    colorlog

conda activate "$ENV_NAME"

echo
echo "Verifying the install..."
python - <<'PYTHON'
import importlib

required = ['numpy', 'scipy', 'kwant', 'tinyarray', 'h5py', 'tomli', 'colorlog', 'matplotlib']
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f'  {name:<12} {getattr(module, "__version__", "(no __version__)")}')
    except ImportError as exc:
        missing.append(name)
        print(f'  {name:<12} MISSING ({exc})')

if missing:
    raise SystemExit(f'Missing packages: {", ".join(missing)}')

import numpy as np
# A Hermitian eigendecomposition is the hot path of the whole calculation, so check that the
# linked LAPACK actually works before queuing anything
a = np.random.default_rng(0).normal(size=(64, 64)) + 1j * np.random.default_rng(1).normal(size=(64, 64))
w, v = np.linalg.eigh(a + a.conj().T)
assert np.allclose((v * w) @ v.conj().T, a + a.conj().T), 'eigh check failed'
print('\n  eigh check passed')
PYTHON

echo
echo "Done. Use it in your submit scripts with:"
echo "    source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "    conda activate $ENV_NAME"
