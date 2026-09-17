# SDD Eight-to-Twelve Data Bridge

## Result

The geometry interface is implemented and verified on the original 40 SDD
training recordings. This is a data-engineering result, not new model training,
forecasting improvement, source admission or independent confirmation.

The previous motion-to-start experiment found a signal in Hotel-to-ETH but not
the reverse direction. Another classifier sweep on the same people would not
provide the missing transfer evidence. This bridge prepares a separately
registered auxiliary-data experiment without changing the ETH/UCY primary task.

| Evidence | Status |
| --- | --- |
| Original source/split and corrected media manifest | `cached_verified` |
| Per-video past-only indices and interface checks | `fresh_run` |
| Independent full-index verification | `fresh_run` |
| Completed resume with artifact hashes | `fresh_run` verification, cached artifacts reused |
| New auxiliary training and predictive evaluation | `not_run`: sampling/source-role contract pending |
| Full-recording image features or multimodal training | `not_run`: this bridge is geometry-only |

## Coverage

Read 8,005,367 annotation rows from bookstore, coupa, deathCircle, gates and hyang.
Only the original 40 train recordings were opened. Original SDD validation/test
annotations and sealed ETH/UCY roles were not opened. The source manifest retains
historical exposure; these recordings are not a new independent test set.

| Raw-frame stride | Past-only queries | Recording-local pedestrian IDs with queries | Sampled interface queries | Complete / partial / absent future labels in sample |
| --- | ---: | ---: | ---: | --- |
| 1 | 3,045,974 | 3,762 | 2,560 | 2,467 / 50 / 43 |
| 12 | 229,333 | 3,468 | 2,514 | 1,917 / 523 / 74 |

These are overlapping windows, not independent examples. Agent counts cannot be
added across strides. Future-support proportions above describe only evenly
spaced interface samples, not the full population. Some short recordings supply
fewer than 64 eligible queries. The two strides are diagnostic configurations,
not a comparison used to choose a favorable training horizon.

Both configurations observe eight sampled states and request twelve future
states. Forecast offsets are 1..12 raw frames or 12,24,..144 raw frames,
respectively. They are not asserted to represent the same physical duration as
ETH/UCY or each other. Pixel coordinates and unknown effective seconds remain.

## Contract and Checks

- Past-only membership: eight uniformly sampled non-lost positions, pedestrian
  ego at query time. Observed neighbors may be other agent types. Neighbor
  offsets/masks explicitly retain incomplete histories.
- Index membership does not depend on future completeness. Lost or absent future
  states receive separate label masks; rows with no future labels remain inputs
  and must not be counted as zero-error evaluation examples.
- Causal finite differences and coordinate transforms use raw past positions.
  No old cached velocity, timestamp, teacher output, goal endpoint or future
  endpoint enters the 476-column geometry interface.
- Occlusion/generated provenance is preserved separately. This remains an
  offline annotated-history task: an API causality test cannot undo retrospective
  annotation interpolation or establish sensor-as-of causality.
- All 5,074 sampled Torch forwards are finite and exactly equal the CV skip at
  zero initialization. There are zero optimizer updates, no accuracy claim and
  no inference-speed benchmark. RGB and image support are explicitly absent.
- Real future mutation and truncation: 320 checks preserve query membership and
  every input array. Synthetic tests check missing-label gradients and reject
  conflicting identities, invalid strides and unregistered training roles.
- A separate contiguous-run implementation verifies all 3,275,307 saved index
  rows, including tails without next-step source support. It does not reuse the
  adapter's sliding-window index builder.

The 80 private index files occupy 124,482,146 bytes (118.72 MiB). They do not
materialize full episodes. Sum of per-record conversion/check durations is
17.36 seconds, excluding startup and the separate independent verifier. This
is not training time. Native arm64 Torch 2.12.0, NumPy 2.4.6, CPU4/interop1,
single process; per-record receipts and heartbeat support verified resume.

Completed resume reused all 40 receipts, built zero recordings and preserved all
121 index/receipt/main-report hashes. Two focused test commands passed: 16 tests
for adapter/state-support and 25 for adapter/causal-recording/visual-model
integration, with five shared tests. They are not 41 distinct tests; the full
legacy suite was not rerun.

## Reproduce

Use the existing licensed local annotation source and original private manifest.
Missing or modified bound source files are a hard error, not silently downloaded
or replaced. Raw data and private indices are excluded from Git.

```sh
.venv-pytorch/bin/python scripts/prepare_m3w_sdd_step_bridge.py --registration configs/m3w_sdd_step_bridge.json --recording bookstore/video0
.venv-pytorch/bin/python scripts/prepare_m3w_sdd_step_bridge.py --registration configs/m3w_sdd_step_bridge.json
.venv-pytorch/bin/python scripts/verify_m3w_sdd_step_bridge.py --registration configs/m3w_sdd_step_bridge.json
.venv-pytorch/bin/python -m pytest tests/test_m3w_sdd_step_adapter.py tests/test_m3w_causal_recordings.py tests/test_m3w_offline_visual_forecast.py -q
```

Registration SHA256: `cf5f6834cc5973612c2764d8de58fca7bfade6957e8a4d1d6831a24fd1f1369e`.
Main report SHA256: `1970c2a7feacf3f6bfa41ebed507374e05edf145ab9c47dc4cb4c51c1f22faa8`.
Code and contract were committed before the pilot as `eeb0628e`.

## Next Decision and Experiment

SDD use was previously authorized; the open issue is not a blanket permission
question or a runtime problem. The new auxiliary training role and raw sampling
interval materially affect the experiment. The focused question offers a
separate stride12 auxiliary arm, stride1 only, or no auxiliary arm. No reply has
been inferred. The main 8/12 protocol and primary ADE remain fixed.

After the choice, register a matched auxiliary-versus-no-auxiliary study with
fixed source roles, update budget and seeds. Verify image support if visual cues
are used. Do not mix raw-frame durations, use masked auxiliary loss to redefine
the complete-label primary, or open sealed cohorts to select the best variant.
No further same-site threshold/classifier grid is justified while this choice
is pending. Nothing here establishes safer forecasts or submission readiness.
No deployment change, Stage5C, SMC, metric, seconds, true-3D or foundation claim.
