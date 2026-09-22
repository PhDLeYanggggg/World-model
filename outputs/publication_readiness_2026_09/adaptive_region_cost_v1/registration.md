# Adaptive Fitting-Region Test: Fixed Development Registration

## Material Passport

Code experiment; source-only development; prospective registration before this
experiment's real fitting and readout. Prior source scenes have already informed
research design. This is not independent calibration, confirmation or deployment.

## Question and Evidence

The completed frozen-region repair improved strict ADE gain from 3.72892% to
4.09764%, but failed scene/seed easy protection. Harm remained underestimated on
11/12 newly selected fitting regions and 12/12 held-source regions. Improving
the old region did not protect the new one. This motivates, but does not prove,
the hypothesis that lagging fitting emphasis contributes to conditional optimism.

## One Changed Factor

Keep the identical initial fourfold weights from the frozen intermediate head.
After every 500 completed updates, before the next batch, recompute the strict
selection on all fitting causal features using the current head. Apply the same
fourfold emphasis and normalize under the unchanged fitting sampler. Refresh at
500,1000,...,11500: exactly 23 times per head. There is no refresh at step12000.
No future labels decide membership. Complete training labels remain used only
for the ordinary supervised cost objective and training-support normalization.

Unchanged: four source sites coupa/deathCircle/gates/hyang; 33 recordings;
175756 past-eligible queries; seeds17/29/43; source-excluded teachers; 356features;
width128,45954parameters; 12000updates, batch256, AdamWlr.001/wd.0001, gradclip5;
squared cost error divided by forecast disagreement; same draws and preprocessing.
Eight observed/twelve predicted annotation steps, stride12, annotation pixels.
Raw-frame t50 is separate, not evaluated as the primary here.

## Readout and Failure Rule

Primary comparison: adaptive strict policy versus the completed frozen-region
strict policy, not an under-budget control. Both use positive net gain, harm at
most .1*benefit, nonzero forecast disagreement and the fixed past-stop veto.
No threshold, refresh-period or multiplier search; no model selection. Retain
all seeds. Same paired 3000 physical-site resamples and equal-site ADE gain.

Joint empirical development pass requires positive lower bound for the paired
ADE-gain difference, each seed positive against causal CV, aggregate AND every
scene/seed positive-easy degradation <=2%, and zero harmed complete exact-zero-CV
queries. Missing futures remain unknown. Passing would not prove population safety.

Secondary: fixed net-stop and legacy matched-intervention-count policies, FDE,
hard/easy slices, every scene/seed, cost bias/MSE on old and new selection regions,
unknown/incomplete support and partial-outcome gain bounds. Strict gain without
equal-count gain is not evidence of improved ranking. Report collapse into near
abstention or unstable region membership; do not promote a secondary winner.

## Execution and Reproduction

Native arm64 .venv-pytorch; CPU4/inter-op1, workers0, no resource probing.
Checkpoints every500, heartbeats every100, atomic resume and runner lock. Record
every refresh's head snapshot and selection for independent-formula replay.
Additional full-fitting inference is counted in compute, not an equal-FLOP claim.
First use synthetic resume/draw checks and real preflight, then a100-update pilot
resumed into the fixed twelve-fit matrix. Fit-only real progress does not open
original val/test, main/external/bookstore roles or new data sources.

Commands with .venv-pytorch/bin/python:

```text
scripts/run_m3w_adaptive_region_cost.py --audit-only
scripts/run_m3w_adaptive_region_cost.py --view <registered-view> --stop-at 100
scripts/run_m3w_adaptive_region_cost.py --resume
scripts/run_m3w_adaptive_region_cost.py --evaluate
scripts/run_m3w_adaptive_region_cost.py --verify
scripts/verify_m3w_adaptive_region_cost.py
```

Independent calibration roles/acquisition decisions remain pending. Selection
is based on offline annotated positions, not guaranteed sensor-time observations.
No metric/seconds/true3D/foundation or deployment claim. Stage5C and SMC remain off.
No new literature novelty claim is made for periodically refreshed weighting.
