#!/bin/bash
#SBATCH --job-name=m3w_easy_cpu_runtime_v1
#SBATCH --partition=cpu
#SBATCH --account=kcl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runtime-%j.out
#SBATCH --error=runtime-%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
test -f .m3w_runtime_owner.json
module load python/3.11.6-gcc-13.2.0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PIP_NO_CACHE_DIR=1
unset PYTHONPATH
if [ ! -d venv ]; then
    python -m venv venv
fi
venv/bin/python -m pip install --disable-pip-version-check torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m pip install --disable-pip-version-check numpy==2.4.6 scipy==1.17.1 scikit-learn==1.8.0 pytest
venv/bin/python -m pip freeze > environment-freeze.txt
venv/bin/python probe_m3w_create_torch.py --output runtime_receipt.json
