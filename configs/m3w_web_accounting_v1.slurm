#!/bin/bash -l
#SBATCH --job-name=m3w_web_accounting_v1
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
#SBATCH --output=m3w-web-accounting-%j.out
#SBATCH --error=m3w-web-accounting-%j.err
set -euo pipefail
module load python/3.11.6-gcc-13.2.0
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec python3 -B -s collect_m3w_create_web_accounting.py collect
