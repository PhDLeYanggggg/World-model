# v10 anonymous reviewer replay

This package accompanies v180. It preserves the v9 checks and adds two
OpenDD vehicle development panels. The paired intervals average donor
directions and seeds before 3,000 location-level draws. The panels do
not include trajectories or checkpoints.

Build from the project root with
`.venv-pytorch/bin/python scripts/build_m3w_vehicle_direction_supplement.py`.
Extract `data/stage_cvpr2027_experiments/submission_admin/review_supplement_v10.zip`
into an empty directory and run `python3 -I -B -S verify.py` there.
Two builds produced identical archives, and isolated replay matched
the original readouts. See the [receipt](receipt.json) for the hashes.
No model was trained or promoted for this package, and it was not uploaded.
