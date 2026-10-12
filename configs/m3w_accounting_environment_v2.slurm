#!/bin/bash -l
#SBATCH --job-name=m3w_account_env_v2
#SBATCH --clusters=create
#SBATCH --account=kcl
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --time-min=00:05:00
#SBATCH --no-requeue
#SBATCH --output=m3w-account-env-%j.out
#SBATCH --error=m3w-account-env-%j.err
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT LD_LIBRARY_PATH LD_PRELOAD
/usr/bin/python3 -I -B probe_m3w_create_accounting_environment.py system
module load python/3.11.6-gcc-13.2.0
exec python3 -B -s probe_m3w_create_accounting_environment.py module
