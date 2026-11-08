# Anonymous OpenDD Scalar Replay v2

The extended archive replays eight exposed OpenDD readouts, including the
component guard, CV difficulty, double-excluded candidate advantage and
the exact-count capacity diagnosis. It contains no raw trajectories,
checkpoint weights, site identifiers or workstation paths.

```sh
.venv-pytorch/bin/python scripts/build_m3w_opendd_candidate_replay.py --extended
```

Local archive:
`data/stage_cvpr2027_experiments/opendd_candidate_replay_v2/anonymous_opendd_replay.zip`.
SHA-256:
`c9556232749fab2aadfb1d4a21a3c0f57fe87129cff3e70dab791cbd3b60864d`.
From a clean extraction, `python replay.py` verifies 48 intervals and
39 count tables. This is scalar replay, not training reproduction or
independent confirmation. The v1 build remains byte-identical.
