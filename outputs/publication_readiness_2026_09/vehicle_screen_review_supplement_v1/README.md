# v13 anonymous reviewer replay

The v183 companion retains v12 and adds the frozen recording-screen
counts and location intervals. It uses seven anonymous location
aggregates and 3,000 paired draws, not trajectories, identifiers or
weights. Two archive builds matched; isolated standard-library
replay reproduced the grouped evidence.

Build with `.venv-pytorch/bin/python scripts/build_m3w_vehicle_screen_supplement.py`.
Extract `data/stage_cvpr2027_experiments/submission_admin/review_supplement_v13.zip`
and run `python3 -I -B -S verify.py` from the extracted directory.
[Receipt](receipt.json). No upload.
