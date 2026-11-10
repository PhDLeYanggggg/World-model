# v11 anonymous replay

The v181 companion retains the v10 checks and adds the vehicle
spacing and learned-forecast comparison. Seven anonymous location
means and 3,000 paired location draws reproduce the intervals.
Grouped scalar counts reproduce the intervention and pass totals.
There are no trajectories or weights. Two archive builds matched
byte for byte, and isolated standard-library replay passed.

Build with `.venv-pytorch/bin/python scripts/build_m3w_vehicle_context_supplement.py`.
Extract the local
`data/stage_cvpr2027_experiments/submission_admin/review_supplement_v11.zip`
to an empty directory and run `python3 -I -B -S verify.py` there.
[Hash and companion receipt](receipt.json). No upload was made.
