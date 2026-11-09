# AV2: A City-Excluded Pair Refit Misses the Useful Slates

I tested whether the expanded acceleration menu needed a selector
trained on better-annotated trajectories. The candidate paths and
intervention budget were held fixed so that this was a test of
selection, not a new forecast producer.

## Comparison

The experiment reuses the 128 AV2 publisher-train scenarios, with
eight observation steps and twelve prediction steps in native units.
No official validation or test files were read. Each of six city folds
uses four cities for fitting, the next city in sorted order for
calibration and the remaining city for scoring. Only donor cities
define the easy and hard reference cuts. These are already exposed
development scenarios, not independent confirmation. The original
CV and OLS4 forecasts, legal candidate paths, two peak offsets and
expanded additive and joint controls are unchanged. Singleton
forecasts stay frozen. Three seeds (17, 29, 43) give 18 actual fits.

The causal head predicts a signed full-grid lower gain relative to
the frozen additive path, whether a candidate meets all utility and
risk checks, and whether it has edited-easy support. It sees only the
history and candidate/reference geometry. Each donor scenario has
equal sampling weight; candidate options are sampled within scenario.
The deployed score in this experiment multiplies the strict
probability by the positive part of the **unconditional** predicted
lower gain. When no score is positive, the forced comparison keeps
the frozen joint path; the calibrated comparison returns CV unless
the score clears a nonempty supported threshold. Calibration retains
all original 2% checks and requires both arms to beat CV and OLS4.
Scenario-view checks are not a recording-level external certificate.

## Results

| Policy | Known ADE | Available FDE | Full-grid CV gain (pp) | Hard CV gain (%) | Risk passes / edited views | Undefined edited easy |
|---|---:|---:|---:|---:|---:|---:|
| CV | 0.25617 | 0.57731 | 0 | 0 | No edits | No edits |
| OLS4 | 0.30431 | 0.65016 | -29.30 | -14.42 | Not matched | Not matched |
| Expanded additive | 0.26175 | 0.58323 | -1.88 | -0.40 | 5/285 | 252 |
| Expanded frozen joint | 0.26084 | 0.58323 | -1.80 | -0.09 | 8/285 | 242 |
| Expanded refit joint | 0.26371 | 0.59152 | -2.79 | -2.14 | 6/285 | 247 |

All three edited policies make 459 edits. Known ADE is over agents
with at least one observed future step; it differs from the previous
complete-track ADE. Risk and hard counts here use the fold's four
donor cities, not the five other cities used for scoring in the
frozen-policy report. Signed easy degradation exceeds 2% in 29
frozen and 27 refit views. City-weighted mean scenario p95 ADE rises
from 0.71375 to 0.73083. The refit's conservative CV gain has a
95% interval of [-4.35, -0.80] pp. Of 174 two-slot seed views, 47
have a strictly feasible candidate against the fixed expanded
additive control, including 32 of 81 complete views. Neither frozen
nor refit selection captures any of the 47. These are view counts,
not independent scenarios. All 32 nonempty calibration breakpoints
fail; the released policy makes zero edits.

| Direct comparison, refit minus frozen | Pair gain in native units | 95% interval |
|---|---:|---:|
| Co-observed pair steps, all scenarios | -0.006239 | [-0.011838, -0.000598] |
| Full grid, all scenarios | -0.008817 | [-0.016804, -0.001610] |
| Full grid, 59 complete scenarios | -0.015051 | [-0.027668, -0.002430] |

The paired bootstrap uses 3,000 draws over six cities, averaging
seeds within scenario before aggregating. Complete labels do not
rescue the refit. Across folds the training roster contains 34-43
two-slot scenarios; only about 0.7-2.1% of their legal assignments
qualify. Large candidate counts do not create independent examples.

## Where Selection Fails

The read-only rank audit recomputes every score without changing
the trained model, thresholds or predictions. On held cities:

- A qualifying candidate survives the positive-gain requirement in
  only **3/47** oracle-positive views. In **39/47**, no candidate
  of any kind receives positive predicted gain.
- Strict probability alone places a qualifying candidate first in
  **4/47** views and in the top five in **18/47**. The actual product
  score puts one in the top five in only **1/47** and first in none.
- The first ranked positive-score action is active in 33 of 174
  views and has edited-easy support in three. Calibration is not the
  first point of failure.

This does not justify deploying the probability-only variant.
It motivates a specific correction: predict expected gain
*conditional on a candidate qualifying* and multiply by qualification
probability, or directly regress the supported gain with a zero
outcome on nonqualifying candidates. The current product mixes a
strict probability with a gain averaged over both good and bad
outcomes. A hard sign test on that average removes good choices.
The next comparison should retain the exact menu and original risk
rule and test these distinct targets on donor/calibration data first.
A larger development roster and a matched public joint forecaster
remain necessary to decide how general the finding is.

## Reproduce

With the [expanded AV2 forecasts](../av2_approach_menu_v1/README.md):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_pair_refit.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_pair_refit.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_pair_refit_ranks.py
```

Four Torch threads, zero data workers, native arm64. Training and
scoring took 16.61 s and 4.35 s respectively, below 700 MB RSS.
Each fit has an atomic checkpoint every 40 updates and a heartbeat.
Repeating `fit` resumes a partial fold or verifies its completed
receipt. Twenty-one focused tests passed. No CREATE job, Stage5C,
SMC or reserved confirmation access. No deployment or manuscript
change. [Registration](registration.json), [freeze](freeze.json),
[metrics](summary.json), [rank attribution](rank_attribution.json).
