# Anonymous replay companion for v179

The v7 archive retains the v6 frontier data and all earlier evidence,
without adding another nested archive. The new part recomputes exact
counts and same-view chance contrasts from anonymized aggregates.
It does not need trajectories, checkpoints, or external Python packages.

Local archive: `data/stage_cvpr2027_experiments/submission_admin/review_supplement_v7.zip`.
Extract into an empty directory and run `python3 -I -B -S verify.py`.
The [receipt](receipt.json) binds the archive, paper and exact readout.
Deterministic build and clean replay passed. Nothing was uploaded.
