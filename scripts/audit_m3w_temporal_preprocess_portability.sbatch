#!/bin/bash -l
#SBATCH --job-name=m3w_temporal_preprocess_audit
#SBATCH --account=kcl
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/users/k24101830/m3w/european_temporal_auxiliary_v1/preprocess-audit-%j.out
#SBATCH --error=/users/k24101830/m3w/european_temporal_auxiliary_v1/preprocess-audit-%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
module load python/3.11.6-gcc-13.2.0
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd /users/k24101830/m3w/european_temporal_auxiliary_v1/code
exec /users/k24101830/m3w/easy_hurdle_runtime_v2/venv/bin/python -m scripts.audit_m3w_temporal_preprocess_portability --root /users/k24101830/m3w/european_temporal_auxiliary_v1
