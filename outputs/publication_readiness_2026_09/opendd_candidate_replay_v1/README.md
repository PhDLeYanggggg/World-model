# Anonymous OpenDD Scalar Replay

This package covers four completed development studies: donor shape
generation, learned shape choice, resolved-error recipient choice,
and the compact EqMotion comparison. It replays the exact counts
and descriptive intervals, not model fitting or raw trajectory error
calculation. The archive contains no identities, paths, data or weights.

```sh
.venv-pytorch/bin/python scripts/build_m3w_opendd_candidate_replay.py
```

Local archive:
`data/stage_cvpr2027_experiments/opendd_candidate_replay_v1/anonymous_opendd_replay.zip`.
SHA-256:
`3084841a70c07e879786f04cc47c3891a25be17f4ea8d98f716392a198c8e332`.
From a clean extraction, `python replay.py` verified 31 intervals and
20 count tables. The required NumPy version is in the archive.
This is evidence replay, not independent confirmation.
