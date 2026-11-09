# AV2: Most Matched Timing Opportunities Do Not Require Reassignment

The [temporal refit](../av2_temporal_refit_v1/README.md)
recovered some references on a wider bank but all held
releases failed. Instead of replacing the old independent
reference, I keep its selected recipients, exact peak
displacements and original path. Each edited trajectory
can use its original, early or late timing. This yields up
to eight distinct substitutions plus the incumbent.

## Comparison

- **Replay only:** prefer a substitution when its shape,
  aligned to earlier observed prefixes, improves both
  individual and pair error over the incumbent in replay.
  These are retrospective checks on the observed eight
  steps, not claims that the final-origin template was
  known at those earlier times.
- **Geometry head:** fit dense relative margins and exact
  matched qualification with all replay channels masked.
- **History head:** same inputs plus the replay channels;
  matched architecture, initialization and sampled updates.

Train on four donor cities. Reuse 72 inner action
checkpoints and city-excluded predictor inputs to build
4,776 donor query views and 38,208 alternatives, including
540 matched positives. The calibration city and held city
are excluded from that training. The two heads share
320 updates and a batch size of 128. All alternatives and
thresholds are hashed and sealed before held scoring.

Only substitution queries passing the original calibration
can release. Both arms must beat CV and OLS4, pass the
three 2% tests with supported denominators, and the
candidate must improve pair error without increasing
trajectory error. In the incremental diagnostic no switch
means retaining the incumbent. In the original matched
release comparison both arms instead use CV outside the
release prefix; the direct between-arm gain is identical
because both arms agree on those views. Neither diagnostic
certifies deployment of the incumbent or of a substitution.

## Where the Capacity Comes From

1,020 two-slot seed views, 340 scenarios, six development
cities and three seeds. Only 63 fixed references qualify.
The following counts are exact hindsight witnesses:

| Alternatives allowed against the same fixed reference | Views with a qualifying alternative |
|---|---:|
| Timing only; same forecast sources, actors and peaks | 51 |
| Any source; same actors and assigned peaks | 56 |
| Any source and either peak assignment to the same actors | 56 |
| Full union, any recipients | 61 |

The local menu evaluates 8,160 nonidentical alternatives
across all views. It covers **51/61** of the full union's
fixed-reference opportunities. This is not a statement
that most scenes benefit: 51 is 5.0% of all 1,020 views.
Palo Alto still has zero eligible fixed references.
Independent scalar checks reproduce each qualifying
capacity witness and every local alternative label.

## What Past-Only Selection Recovers

| Method | Calibration top-ranked / 49 available | Held top-ranked / 51 available | Passing calibration selections | Held pass / released |
|---|---:|---:|---:|---:|
| Replay only | 27 | 29 | 0 | No release |
| Geometry | 23 | 24 | 4 | 0 / 4 |
| History (primary) | 27 | 24 | 5 | 1 / 5 |

The paired city-balanced history-minus-geometry top-1
change over all views is -0.02 pp [-0.50, +0.49]. The
simple replay rule retrieves more qualified options than
the fitted heads before calibration but cannot be released.
The primary's one matched success is Washington DC,
seed 43: trajectory lower gain +2.108 pp, pair lower
gain +0.001478 native units, and signed easy degradation
-2.58%. In Austin, three of four releases have ineligible
references; the remaining view fails both relative gains.

Across all views, the primary's released lower trajectory
gain is +0.00315 pp [0, +0.00901] and pair gain is
+0.00000171 [-0.00000104, +0.00000616] native units. The
3,000 draws average seeds within scenario before resampling
the six cities. The evidence supports a compact candidate
set and one qualifying case, not robust improvement.
The next bottleneck is recognizing when *both* paths qualify,
not expanding the menu further. Keep replay as the control
and test a separate donor-only reference-eligibility estimate
before retraining the relative selector.

## Reproduction and Supplement

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_incumbent_timing.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_incumbent_timing.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_incumbent_timing.py
.venv-pytorch/bin/python scripts/plot_m3w_av2_incumbent_timing.py
.venv-pytorch/bin/python scripts/package_m3w_av2_incumbent_timing.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py \
    tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

Native arm64, four Torch threads, one process, 36 fresh
head fits, 72 focused tests. Fit/freeze 49.25 s at 0.67 GB
maximum RSS; score 7.88 s; capacity audit 4.41 s.
Checkpoints every 40 updates store optimizer and sampler
state. Repeat `fit` to resume. Files stay under
`data/stage_cvpr2027_experiments/av2_incumbent_timing_v1`.

The local `submission_admin/av2_incumbent_timing.zip`
contains a checked vector figure, anonymous group totals
and a standard-library verifier. In an extracted copy,
`python -I -B -S verify.py` recomputes totals and release
checks from stored scalar evidence, including the already
verified incumbent-eligibility flag. It does not replay
training or raw forecasts. Two independent exports matched
byte-for-byte; ZIP SHA-256:
`7f4a0de3bb8a639925520342ac0dd3c321d6d0feb4e4f9fcd099378023fc841b`.

No CREATE submission, new data, reserved confirmation
outcomes, manuscript edit or deployment change.

[Registration](registration.json) · [Freeze](freeze.json) · [Results](summary.json) · [Capacity attribution](capacity_attribution.json)
