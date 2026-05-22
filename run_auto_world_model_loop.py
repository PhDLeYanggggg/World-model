from __future__ import annotations

import os


# Apple Silicon/local OpenMP guard. These must be set before importing NumPy,
# Torch, SciPy, or any project module that may transitively load BLAS/OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.orchestrator.auto_loop import main


if __name__ == "__main__":
    main()
