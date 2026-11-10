# v15 anonymous reviewer replay

The v185 companion retains the v14 evidence and adds fixed-action
family alignment. It uses seven anonymous location aggregates
and 3,000 draws, with no raw trajectories or trained weights.
Two builds produced identical archives, and isolated
standard-library replay matched the original readout.

Build with `.venv-pytorch/bin/python scripts/build_m3w_vehicle_family_supplement.py`.
Extract `data/stage_cvpr2027_experiments/submission_admin/review_supplement_v15.zip`
and run `python3 -I -B -S verify.py` inside the extracted directory.
[Receipt](receipt.json). No upload was made.
