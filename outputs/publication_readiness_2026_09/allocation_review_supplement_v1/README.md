# Anonymous replay companion for v176

The new supplement retains the verified v3 archive and adds the two OpenDD
allocation controls: the fixed-count movement ceiling and the fitted head
comparison. It verifies the old evidence before recomputing the new tables
and location intervals. The scalar export contains no checkpoint, raw label,
recording identifier or path to local data.

The local archive is `data/stage_cvpr2027_experiments/submission_admin/review_supplement_v4.zip`.
Extract it and run `python3 -I -B -S verify.py`. The clean standard-library
replay and a byte-for-byte rebuild check passed. The archive remains local;
only its generator and [hash receipt](receipt.json) are tracked. Nothing was
uploaded to the submission portal.
