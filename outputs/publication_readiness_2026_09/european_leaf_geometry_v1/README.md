# Fixed-Routing Geometric Leaf Training

## Material Passport

Complete paired training control; advancement failed, no deployment. Existing
data/models are cached-verified; terminal values and readout are fresh-run.
See [conclusions](conclusions.md), [failure analysis](failure_analysis.md),
[protocol](protocol.md) and [next action](next_action.md).

## Reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest tests/test_m3w_leaf_geometry.py tests/test_m3w_leaf_geometry_readout.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_leaf_geometry.py
```

On a fresh registered result directory with exact prior assets, the runner
supports `register`, `pilot`, `run` and `run --resume`. Incomplete resumption
refits/verifies existing groups and rejects immutable differences. Completed
runs refuse overwrite. Source registration commit: `709ede75`.

The 72 diagnostic checkpoints are in owned CREATE storage, identified by
`checkpoint_manifest.json`, not Git. Each stores terminal fractions, offsets,
leaf support statistics and identity. Inference also needs the hash-bound
original forest/preprocessing; `m3w_leaf_geometry.predict` composes them.
These are not approved deployment checkpoints.

## Runtime and Recovery

- Native arm64, Torch2.12.0, CPU4/interop1, workers0. Terminal fitting uses
  closed-form weighted means, not Torch gradient training.
- Pilot PID27499 completed; checkpoint 2,844,372bytes verified remotely.
- First full attempt exited1 before fitting because a new SSH stream was
  rejected with an MFA/public-key message. A bounded read-only retry on the same
  authorized connection succeeded. Unchanged code/config resumed successfully;
  no credentials, host checks or authentication policy were changed.
- Successful PID28012 exited0. Last group completed at177.1361s; remote checks
  followed, so this excludes final verification time. Fit plus exact refit:
  79.8185s. Inference/readout:41.4631s. PeakRSS:9,634,021,376bytes.
- 72 checkpoints,42,697,196bytes, all remote hashes verified. No local numeric
  cache. Existing tree counts give an88,375,200byte raw-array allowance, below
  the128MiB cap. Personal remote quota remains unknown, not assumed unlimited.
- Local group aggregates total672,311bytes. Local disk stayed below the unchanged
  10GiB cache reserve; new numerical arrays stayed off local disk.
- No new Slurm job or remote science process. Login-side activity is file I/O
  and checksum verification only. Existing simulation work was untouched.

## Verification

15 scoped tests pass. Original prediction/action hashes, training input hashes
and reference outputs match. Original means in912,895 populated leaves are
reconstructed; zero target clamps. All72 refits, serialized checkpoints,
predictions and readouts replay exactly. 8,280 scalar/parent and2,050 independent
aggregate fields checked. Model replay uses the same implementation, not an
independent model. Full historical suite and independent evaluation were not run.

Scope: 12 exposed development localities, obs8/pred12 raw stride12,
image-local detector-silver. No metric/seconds, human-gold, physical safety,
true3D, foundation or CVPR-readiness claim. Stage5C/SMC off.
