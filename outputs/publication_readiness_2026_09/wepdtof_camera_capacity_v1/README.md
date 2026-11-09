# Why the camera policy could not catch OLS4

The original small CV-centered edits cannot catch OLS4 on the
exposed Convenience Store camera, even when given hindsight.
This isolates a limit of the allowed actions, not just a failure
to pick the right person. It does not establish the same limit
on the other cameras, whose outcomes were not reopened.

| Donor seed | Fixed edits | Sum of peak shifts (px) | Arbitrary upper vs OLS4, known steps (pp) | Arbitrary upper vs OLS4, conservative (pp) | Expanded finite menu with 2% risk, joint LP upper (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 23 | 73.14 | -18.56 | -19.39 | -22.91 |
| 29 | 23 | 68.93 | -18.99 | -19.83 | -23.18 |
| 43 | 23 | 77.22 | -18.18 | -19.02 | -22.70 |

The three seeds share one camera and 58 online track-windows, of
which 57 have complete futures. The upper bounds allow arbitrary
paths at each query's original radius and up to its original count;
eligible recipients are unchanged. Arbitrary bounds ignore risk,
and the joint finite-menu LP enforces the original risk constraints
but relaxes integer choices. Both original and expanded menus are
excluded in all three cases. A negative observed-only bound shows
that missing futures are not sufficient to explain this gap.
Gain points use the common known CV error denominator, not the
percentage change relative to OLS4.

## What changed on real camera histories

- The past-prefix ranker using the four linear alternatives and
  the same edit counts/radii passes the original risk check in
  3/3 seed views. Conservative gain is +1.26 to +1.43 pp over CV
  and -25.12 to -24.95 pp over OLS4. It passes **0/3** joint checks.
- Without the small-edit restriction, using the componentwise
  median pair slope anchored to the last detection reduces known
  ADE from **14.61 px (OLS4) to 13.95 px**, with **+3.03 pp**
  conservative gain against OLS4. Its easy positive-harm upper
  bound is **22.89%**, so it fails the original 2% test.
- Replacing the last position with the fitted line endpoint
  does not improve known ADE for either slope. On this camera,
  the useful signal is robust velocity estimation, not the tested
  intercept correction. This is a comparison of four fixed
  estimators, not evidence that camera jitter caused the gap.

Do not train a larger selector for these same tiny actions. The
next model experiment should improve the camera-history forecast
itself, then test whether it can preserve the original CV-referenced
risk without needing near-total abstention. Robust slope is a
development control for that experiment, not the new deployment.
The previously scored external groups cannot be a new confirmation
set. No original risk threshold or result changed.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_camera_capacity.py
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_camera_capacity.py --replay
```

Fresh calculation: causal controls, arbitrary and finite bounds on
the development pilot. Cached and hash-checked: camera histories,
pilot cutoffs, donor checkpoints, and the original fixed-count
result. Local native arm64 Torch, four compute threads, no
workers or CREATE submission. Each seed is written atomically;
the replay recomputes the solver bounds and checks the receipts.
Twelve focused tests passed. No new model training, external
scoring, Stage5C or SMC. Eight observed and twelve predicted
sampled steps; pixel coordinates only.

[Registration](registration.json) | [Readout](readout.json) |
[Working manuscript v160](../evidence_manuscript_v160/README.md)
