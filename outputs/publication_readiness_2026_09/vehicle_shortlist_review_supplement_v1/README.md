# Anonymous aggregate replay v17

The v187 companion archive is
`data/stage_cvpr2027_experiments/submission_admin/review_supplement_v17.zip`.
It retains the earlier panels and adds the eight shortlist arms, the
original-mask hindsight caps, and the paired seven-location bootstrap.
The [receipt](receipt.json) binds the archive, manuscript and readouts.

Extract into an empty directory and run `python3 -I -B -S verify.py`.
The archive contains no source trajectories or model weights. It verifies
aggregation and intervals, not model training or external generalization.
