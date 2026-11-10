# Anonymous aggregate replay v19

The v189 companion is
`data/stage_cvpr2027_experiments/submission_admin/review_supplement_v19.zip`.
It preserves v18 unchanged and adds the direct-versus-anchor-difference
training comparison. The [receipt](receipt.json) binds the replay to the
manuscript and frozen evidence. Extract into an empty directory and run
`python3 -I -B -S verify.py`.

This reaggregates the fixed-mask choices and location bootstrap. It
does not include raw data or checkpoints and does not retrain the models.
