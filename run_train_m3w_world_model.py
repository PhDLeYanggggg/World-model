from __future__ import annotations

import os
import platform
import sys


# Set runtime thread defaults before importing NumPy or Torch through the
# backend module. Users can override these in the shell for larger machines.
os.environ.setdefault("WORLD_MODEL_TORCH_THREADS", "4")
os.environ.setdefault("WORLD_MODEL_TORCH_INTEROP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if (
    sys.platform == "darwin"
    and platform.machine().lower() == "x86_64"
    and os.environ.get("WORLD_MODEL_ALLOW_RISKY_OPENMP") != "1"
):
    raise SystemExit(
        "Refusing torch training under macOS x86_64/Rosetta because this runtime "
        "can trigger Intel OpenMP Can't open SHM hangs before Python can recover. "
        "Run with .venv-pytorch/bin/python on arm64, or set WORLD_MODEL_ALLOW_RISKY_OPENMP=1 "
        "only if you accept the crash risk."
    )

from src.m3w.train import main


if __name__ == "__main__":
    main()
