# v12 anonymous reviewer replay

This companion to v182 preserves the previous replay and adds the
action-value counts and paired location intervals. The seven
location aggregates and 3,000 draws contain no trajectories,
identifiers or weights. Two archive builds matched byte for byte;
isolated replay matched the frozen report.

Build with `.venv-pytorch/bin/python scripts/build_m3w_vehicle_action_supplement.py`.
Extract the local
`data/stage_cvpr2027_experiments/submission_admin/review_supplement_v12.zip`
and run `python3 -I -B -S verify.py` inside the extracted directory.
[Receipt](receipt.json).
