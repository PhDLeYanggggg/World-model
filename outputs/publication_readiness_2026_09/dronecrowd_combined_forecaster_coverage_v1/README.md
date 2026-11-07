# The Complete Forecaster Library Has Source-Screen Coverage

The recent 48-recording comparison covered 12 of my 18 saved PCT and EqMotion
checkpoints. The other six were not covered by that particular screen because
10 contributing training recordings did not occur in its A/B image roster.
The earlier full MovingDroneCrowd++ screen does include all ten, with three
source views per recording against the same 112 DroneCrowd clips and 45 groups.
Joining the two screens closes this training-roster coverage gap without
rerunning the forecast training or downloading any more images.

- `cached_verified`: the full 20,580 original pair receipts and grouping match
  the archived result; this includes all 13,440 cross-archive pairs. The new
  read-only recheck re-extracts 360 prepared descriptors and repeats the same
  fixed 50 pairs, not all original matching. Exit 0 in 42.45 s.
- `fresh_run`: actual training-roster join for all 18 predictors; exact source
  ID, external grouping and original audit hashes. The additional 1,120 saved
  cross-archive pairs for the ten recordings are all unsupported by the
  archived match thresholds, with no ambiguous or strong candidates.
- `not_run`: no assignment of scientific roles, no forecast or future-label
  readout on the 112 clips, no learned cost-head admission, no proof of physical
  independence. The coverage statement is about existing predictors only.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/check_m3w_movingdronecrowd_background.py
.venv-pytorch/bin/python scripts/check_m3w_dronecrowd_combined_coverage.py
```

The original narrow-screen record stays intact. Neither screen proves there
are no shared locations or supplies a scientific admission decision. Full
external evaluation still needs the separate role freeze. No Stage5C or SMC.

[Joined coverage](analysis.json) | [Read-only recheck](broad_readonly_recheck.json)
