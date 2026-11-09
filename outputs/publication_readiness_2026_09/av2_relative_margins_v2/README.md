# AV2: Learning Margins Against the Actual Independent Choice

The [easy-gate experiment](../av2_easy_gate_v1/README.md)
had one held edit that passed its own risk checks, but it
lost to the independent choice at the same budget. Here I
train directly on that comparison, instead of treating
standalone qualification as a proxy for relative value.

## Matched Experiment

Every arm starts from the same city-excluded EqMotion
forecast bank, the same fixed independent additive choice
and the same two peak edit magnitudes. A copy of the
reference is not a new candidate. Both new arms receive
the same history and forecast geometry, including the
feature difference from the actual independent choice.
They share architecture, initialization, optimizer, update
count, and sampled options. Both sample one positive and
one negative with edited-easy support and an eligible
independent choice per donor query when present.

- **Binary:** predict exact matched qualification.
- **Dense:** predict that same event, plus comparator and
  edited-easy support, pair-relative gain, trajectory-relative
  gain, the worse motion-floor margin and the tightest of the
  three risk margins. Relative gains debit unobserved steps;
  the risk margin is trained only where edited-easy support exists.

All inputs are observed history or candidate-path geometry.
The outcome supplies only training targets, calibration labels
or evaluation labels. The scale is the history-based scene
motion unit used by the earlier paired control. Scores are
not probabilities of safety. On each calibration city I
check all distinct nonempty score prefixes against the
unchanged rules: both arms pass CV and OLS4, all three
2% risks are defined and pass, and the joint path must
improve pair error without increasing trajectory error.
Both arms return CV outside the selected prefix. Forecast
rankings and thresholds are sealed before held scoring.

## Correcting the Vectorized Labels

The first preflight stopped on a nonfinite all-harm margin
before fitting. The legacy vectorized expression had
multiplied an undefined ADE by an edit indicator, which
still yields a NaN when a wholly unlabelled actor is
unchanged. The scalar evaluator already excluded unknown
actors from known harm and charged an unknown actor's
maximum disagreement only when edited. I use the same
rule in the new vector targets. A missing easy reference
still cannot pass. I preserve and replay the legacy counts
separately, and verify differing labels and selected options
against the scalar evaluator. This is an implementation
correction, not a change to the evaluation protocol.

## Results

The 18 fold-seed fits each train two heads for 320 updates.
They reuse 72 city-excluded additive models and their
forecast producers. Of 4,776 donor seed views, only 164
contain any matched positive option. Held scoring has 1,020
two-slot views; the fixed independent choice passes in 63.
Only 48 of those views have any qualifying joint option.
The label correction changes no matched positive in this
fixed-anchor calibration or held sample. It still matters
for the dense targets and for the earlier unrestricted
capacity calculation: 63 held views, comprising 22,834
candidate evaluations, contain a wholly missing actor.

| Arm | Calibration top 1 / 44 available | Held top 1 / 48 available | Held top 5 / 48 available | Held pass / released |
|---|---:|---:|---:|---:|
| Prior direct ranker | 6 | 9 | 14 | No release |
| Matched binary | 5 | 4 | 12 | 0 / 1 |
| Dense margins (primary) | 8 | 5 | 11 | No release |

On the oracle-supported held subset, the paired city
average top-1 change is **+6.39 pp [1.21, 12.68]** for
dense vs the matched binary arm. The interval uses 3,000
draws over the five cities with support, after averaging
within scenario across seeds. This is a small conditional
ranking result, not superiority over the prior ranker.
Both new arms recover fewer top-1 choices than that prior
and neither has a passing held release.

The binary threshold releases once in Palo Alto (seed 17).
The independent choice is ineligible, the new choice
has **20.87%** easy positive harm, and trajectory lower
gain vs that reference is **-0.2745 pp**. The slight pair
gain on that one release does not compensate for these
failures. The dense result equals CV on released utility
because it releases nothing, not because it passed safety.

## Why the Ranking Change Was Not Enough

Among the 48 held views with a lawful improvement, dense
training reduces overall-harm failures from 19 to 14 and
dual-motion-floor failures from 15 to 12 relative to binary
training. But it fails trajectory noninferiority in 37
views (binary: 34) and pair improvement in 34 (binary: 33).
These reasons can overlap. Fewer risk failures are not a
substitute for a better paired trajectory.

| Forced two-slot choice vs independent | Trajectory lower gain, pp [95% CI] | Pair lower gain, native [95% CI] |
|---|---:|---:|
| Prior direct ranker | -3.270 [-4.831, -1.620] | -0.02382 [-0.04163, -0.01220] |
| Matched binary | -1.615 [-2.874, -0.493] | -0.01429 [-0.02763, -0.00566] |
| Dense margins | -0.769 [-1.239, -0.384] | -0.00482 [-0.00728, -0.00228] |

These are six-city summaries, not deployment outcomes.
On the 465 completely labelled views the dense means
are near zero and both intervals include zero. The
remaining gap has two parts: the fixed reference already
disqualifies 957 views, and the model recovers only five
of the remaining 48 available joint choices. This supports
testing how to improve eligible candidate and reference
supply with fixed budgets before fitting another ranker.
No thresholds should be retuned from this readout.

## Reproduction

Use the existing local AV2 publisher-train development
objects and hash-checked receipts from the
[paired qualification experiment](../av2_paired_qualification_v1/README.md).
No new raw data or remote training is needed.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_relative_margins.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_relative_margins.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_relative_margins.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py \
    tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

The trainer runs single-process with four Torch threads.
It saves the network, optimizer and sampler state every
40 steps and writes an atomic heartbeat with its PID.
Repeat `fit` to resume; completed fold receipts and hashes
must match. The registered primary is the dense arm.
Checkpoints and forecasts remain local under
`data/stage_cvpr2027_experiments/av2_relative_margins_v2`.
Fit plus freeze took 213.50 s, with 1.38 GB maximum RSS.
Held scoring took 37.01 s and the failure attribution took
30.80 s; the focused suite passed 63 tests.

The six cities and three seeds use the established 8/12
observation/forecast grid. This is a development replay,
not independent confirmation, and uses native units and
steps. Reserved outcomes, the manuscript and the deployed
policy were not changed.
