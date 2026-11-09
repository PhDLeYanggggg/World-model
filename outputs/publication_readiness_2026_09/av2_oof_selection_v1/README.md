# AV2: City-Excluded Selection and Nonadditive Pair Gain

The [frozen EqMotion comparison](../av2_eqmotion_v1/README.md)
left a gap between available and selected good slates. I
retrained the selection layer on the expanded menu and tested
whether explicit nonadditive pair gain helps at the same edit
budget.

## Nested Forecasts and Matched Decisions

The 1,024 publisher-train UUIDs, 8/12 grid, original donor-only
cuts and six rotating city roles are unchanged. The held
readout uses the same 896 added scenarios, including 422 with
complete roster futures. Coordinates are AV2 native units.

For a given outer held city and separate calibration city,
the selection layer trains on the remaining four cities.
Each of those four receives EqMotion candidates from a model
trained on the *other three*. The predicted donor city, the
outer calibration city and the outer held city are therefore
all excluded from its producer. Identical inner training sets
are reused. There are **54 actual inner EqMotion fits**
(18 triplets, three seeds), and **36 selector fits**
(18 independent action heads and 18 slate heads). The
calibration and held forecasts use the unchanged four-donor
EqMotion weights. The held forecasts are sealed before scoring.
This avoids in-sample producer predictions within each fold;
the rotating cities are still development, not external confirmation.

The new uniform and additive arms share the same independent
action scores, with the uniform arm additionally requiring the
same path family across the two slots. The slate head estimates
qualification probability, conditional smaller gain over CV
and OLS4, and conditional nonadditive pair credit. Its risk-only
and interaction arms use the **same checkpoint**; the only
scoring difference is the pair term. The
[method note](method_note.md) derives the four-path contrast and
its missing-step lower bound. All forced arms use the same
candidate forecasts and 2,952 edit peaks. The original three
2% checks and separate calibration city are unchanged.

## Forced Results

| Policy | Known ADE | Available FDE | Conservative gain vs CV (pp) | Hard gain vs CV (%) | Risk passes / edited views | Risk and both motion floors |
|---|---:|---:|---:|---:|---:|---:|
| CV, no edits | 0.24668 | 0.57458 | 0 | 0 | No edits | No edits |
| OLS4, unmatched | 0.27864 | 0.61665 | -22.03 | -10.91 | 9/2,688 | 0 |
| Raw EqMotion, unmatched | 0.34732 | 0.66486 | -87.46 | -10.42 | 1/2,688 | 0 |
| Frozen scene uniform | 0.24573 | 0.56897 | -1.01 | +0.59 | 100/1,932 | 47 |
| Frozen additive | 0.24582 | 0.56924 | -1.06 | +0.59 | 110/1,932 | 59 |
| Frozen joint | 0.24578 | 0.56899 | -1.04 | +0.57 | 105/1,932 | 52 |
| Refit scene uniform | 0.24631 | 0.56954 | -1.24 | +0.35 | 132/1,932 | 63 |
| Refit additive | 0.24574 | 0.56860 | -1.07 | +0.56 | 143/1,932 | 77 |
| Refit slate, risk only | 0.24860 | 0.57447 | -2.33 | +0.04 | 103/1,932 | 46 |
| Refit slate, with interaction | 0.24855 | 0.57437 | -2.38 | +0.08 | 99/1,932 | 41 |

The frozen controls replay exactly. Relative to frozen additive,
refit additive lowers undefined edited-easy ratios (1,666 to
1,482) and improves the count of qualified views, but more
edits exceed the signed easy limit (118 to 242). The two slate
arms have 358 and 363 signed-easy violations. Their mean
scenario p95 ADE is 0.69303 and 0.69349, above CV's 0.69153;
refit additive is 0.68552. Worst observed scenario-seed p95 ADE
is 5.929 for refit additive, versus 7.432 for CV and interaction.
The improved additive tail does not establish risk-controlled
release or positive conservative mean gain.

| Direct conservative pair contrast | All held views, native gain [95% CI] | Complete scenarios, native gain [95% CI] |
|---|---:|---:|
| Interaction vs risk only, same head | -0.000813 [-0.001694, -0.000200] | -0.000553 [-0.001759, +0.000334] |
| Interaction vs refit additive | -0.010845 [-0.014155, -0.008030] | -0.002597 [-0.005371, -0.000047] |
| Interaction vs refit scene uniform | -0.010099 [-0.013774, -0.007284] | -0.002452 [-0.005333, +0.000314] |
| Risk only vs refit additive | -0.010984 [-0.014209, -0.007986] | -0.002043 [-0.003857, -0.000097] |

Intervals use 3,000 paired city bootstrap draws after averaging
seeds within scenario. On known steps alone, interaction vs
risk only is +0.000075 [-0.000500, +0.000516]. The credit
changes 217 of 1,020 two-slot views without delivering
a reliable benefit. The two slate models are worse than
additive even on fully labelled scenarios; missingness is
not the only reason.

## Why the Policy Does Not Release

Of 1,020 held two-slot views, 293 have a strict joint alternative
against the refit additive choice. Additive passes the dual
motion and risk checks in 63 views; both arms could pass in
48 with hindsight. The actual risk-only and interaction
decoders select 12 and 9 strict alternatives respectively,
but **none** has a qualifying additive control at the same
time. The calibration city sees 12 strict alternatives chosen
by each decoder and no simultaneous pass. All **2,040
nonempty calibration breakpoints** fail (1,020 each); both
matched release arms return CV everywhere. The capacity counts
are scenario-seed views, not independent scenes or release
results.

This establishes the negative result with out-of-city neural
training candidates, not just frozen scores or in-sample
predictor exposure. The additive improvement in support is
worth keeping as a diagnostic, but neither slate score should
be deployed. The next test should ask whether causal
within-history backtesting can rank **within the same query**
for dual gain and easy harm, against the trained reference.
The 48-view exact ceiling makes that a direct test of
decision information rather than another expansion of
candidate capacity. A useful result still needs nonempty
calibration and independent confirmation.

## Reproduce

Use the verified [EqMotion and AV2 assets](../av2_eqmotion_v1/README.md):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_eqmotion_oof.py
.venv-pytorch/bin/python scripts/run_m3w_av2_oof_selection.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_oof_selection.py score
```

Native arm64 and four Torch threads, with no loader workers.
The OOF stage took 707.29 s, selector training and freeze
133.41 s, and scoring 17.68 s. Peak RSS was under 1.25 GB.
Checkpoints every 50 producer or 40 selector updates retain
weights, optimizer and sampler state; rerun the same phase to
resume. The heartbeat records process, fold and step.
Thirty-five focused tests passed. A separate real-input
gradient check used 4,150 candidates in ten donor scenarios.
No CREATE job, reserved confirmation data, Stage5C, SMC,
manuscript or deployment changes. Raw data, per-scenario
outputs and weights remain local.
[OOF roles](../av2_eqmotion_oof_v1/registration.json),
[OOF receipts](../av2_eqmotion_oof_v1/freeze.json),
[selector protocol](registration.json), [freeze](freeze.json),
[metrics](summary.json).
