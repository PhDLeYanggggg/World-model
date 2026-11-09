# Restore the relative positions of observed vehicles

The original vehicle cohort stores each eight-step history relative to
its own last position. That is enough to fit motion, but loses the
spacing between vehicles in a query. I recovered the last observed
positions from the original checked SQLite archives and stored them
relative to the first observed vehicle in each query. Adding this
offset to the centered history reconstructs the shared query frame.

The reconstruction matches all **141,783 rows, 25,596 queries and 500
recordings** in the frozen OpenDD vehicle cohort. The script checks
vehicle order, query counts and equality of every centered past
coordinate to the archived float32 values. It reads timestamp and
identity metadata to reconstruct the old index, then retrieves
coordinate columns only for the selected eight past samples. The
export contains offsets only, not future coordinates, labels or
absolute map positions. The original cohort is unchanged.

This enables a matched geometry/no-geometry ablation. It is an input
artifact, not a result on forecasting or collision prediction. The
direct pair-learning experiment does not use these coordinates.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-pytorch/bin/python scripts/build_m3w_opendd_vehicle_observed_geometry.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_opendd_vehicle_observed_geometry.py
```

[Per-site row counts and hash manifest](manifest.json). The NPZ files
stay local under the corresponding `data/stage_cvpr2027_experiments`
directory.
