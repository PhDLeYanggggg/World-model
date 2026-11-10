# v14 anonymous reviewer replay

The v184 companion preserves the v13 checks and adds the fixed
nominee counts and paired location intervals. It contains only
anonymous aggregate inputs and 3,000 resampling indices for this
panel. No weights or raw trajectories are included. Two archive
builds and an isolated standard-library replay agreed.

Build with `.venv-pytorch/bin/python scripts/build_m3w_vehicle_nominee_supplement.py`.
Extract the local
`data/stage_cvpr2027_experiments/submission_admin/review_supplement_v14.zip`
and run `python3 -I -B -S verify.py` in the extracted directory.
[Receipt](receipt.json). No upload was made.
