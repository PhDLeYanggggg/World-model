# AV2: Observed-History Replay Recovers Some Qualifying Choices

The [city-excluded selector](../av2_oof_selection_v1/README.md)
missed all 48 views in which both arms could qualify. I kept
the same candidate forecasts and additive control and tested
whether recent motion gives the selector useful information
beyond the existing geometry features.

## Controlled Comparison

For each legal two-agent edit, I align its current displacement
shape to four prefixes of the eight observed positions. Prefix
headings rotate the shape, and the prefix-to-current speed
ratio is clipped to [0.25, 4]. When a heading is unavailable,
the shape is left unchanged. The rest of the observed window
provides one to four positions against which to compare the
aligned shape and prefix CV/OLS4. The features summarize
individual gain and harm, short-prefix easy support, and
pair-relative change. The short-prefix easy reference is
scaled by prefix prediction length; it is an input feature,
not a replacement for the original easy or risk definition.
The shape is constructed at the current origin. This is a
compatibility test with known history, not a claim that EqMotion
predicted the observed suffix from a shorter prefix.

Both variants use the same MLP, seeds, initial weights, sampled
examples, expected-gain score and losses. The geometry control
receives zeroes in the new feature channels. Each version learns
absolute dual-floor and risk qualification, a conditional gain
magnitude and an auxiliary support target. Each ranking pair
contains a qualifying and a nonqualifying option from the same
donor window; the logistic rank loss follows the general
[RankNet construction](https://doi.org/10.1145/1102351.1102363).
Historical motion compatibility has also been used to guide
[forecast generation](https://arxiv.org/abs/2503.24272); here the
forecast generator is frozen and compatibility is tested solely
as information for the decision.

The candidate bank uses the same city-excluded EqMotion
predictions and six city roles. Every old additive prediction
replays exactly. The 36 new selector fits use three seeds;
neither the 54 inner nor the 18 outer forecast producers were
retrained. Calibration keeps the original dual-accuracy and
2% risk checks for both arms and requires a direct pair gain
over the same additive path. The held readout uses the same
896 added scenarios and 2,952 forced edits in native coordinates.

## What History Improves

| Held two-slot result | Geometry | History |
|---|---:|---:|
| A strict alternative is in the top five (293 possible views) | 41 | 80 |
| A strict alternative is selected (293 possible views) | 12 | 35 |
| Top five contains a qualifying pair of arms (48 possible views) | 2 | 12 |
| Both selected arms qualify (48 possible views) | 0 | 5 |

The 48 views represent 35 distinct scenarios across seeds.
History changes the selected path in 902 of 1,020 two-slot
views. Retrieval improves, but choosing between many changed
paths also adds risk.

| Forced policy | Known ADE | Available FDE | Conservative gain vs CV (pp) | Hard gain vs CV (%) | Risk passes / 1,932 | Dual gain and risk |
|---|---:|---:|---:|---:|---:|---:|
| CV | 0.24668 | 0.57458 | 0 | 0 | No edits | No edits |
| Scene uniform, unchanged | 0.24631 | 0.56954 | -1.24 | +0.35 | 132 | 63 |
| Additive, unchanged | 0.24574 | 0.56860 | -1.07 | +0.56 | 143 | 77 |
| Prior risk-only slate | 0.24860 | 0.57447 | -2.33 | +0.04 | 103 | 46 |
| Geometry ranking | 0.24825 | 0.57645 | -2.18 | -0.10 | 105 | 47 |
| History ranking | 0.24761 | 0.57486 | -1.60 | +0.44 | 137 | 72 |

The public EqMotion and OLS4 controls are unchanged and
included in the [metrics](summary.json). Relative to geometry,
history reduces undefined edited-easy ratios from 1,434 to
1,317 but increases signed easy-limit violations from 317 to
383. Its mean scenario p95 ADE is 0.69308, against 0.69402
for geometry, 0.69153 for CV and 0.68552 for additive.

## Known-Label Gain Versus Missing-Step Debit

The **direct** history-minus-geometry trajectory contrast uses
one common full-grid denominator:

| Term | Percentage points of observed CV error [95% CI] |
|---|---:|
| Known trajectory gain | +0.605 [+0.106, +1.144] |
| Missing-step disagreement debit | 0.712 [0.520, 0.945] |
| Conservative full-grid gain | -0.107 [-0.728, +0.513] |
| Gain on 422 complete scenarios | +0.922 [+0.374, +1.480] |

Do not subtract the two policies' separate gains against CV
and call the difference a direct conservative gain. The
head-to-head loss under missingness must charge the change
between the two forecasts themselves. The direct full-grid
**pair** gain is **-0.003417 [-0.005391, -0.001813]** native
units. On complete scenarios it is **+0.002075
[+0.000963, +0.003137]**. Against the unchanged additive
arm, history still has negative full-grid pair gain
(-0.009323 [-0.015171, -0.005738]). All intervals use 3,000
paired city bootstrap draws after averaging seeds in scenario.

On calibration there are one and two locally qualifying
two-arm choices for geometry and history, but none belongs to
a passing nonempty score prefix. **0/2,040** distinct nonempty
calibration breakpoints pass. Both release arms return CV;
there is no independent or deployable gain.

I would retain the history signal for another decision test,
not deploy this ranker. More candidate generation is not the
priority. Test whether the gains on observed positions persist
when a separate head learns full-grid disagreement cost and
signed easy harm against the fixed additive reference. The
calibration rule and reserved external evaluation stay fixed.

## Reproduce

Requires the [city-excluded inputs and frozen reference](../av2_oof_selection_v1/README.md).

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_history_rank.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_history_rank.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_history_rank.py
```

Training and freeze took 211.64 s, scoring 18.43 s, with
maximum RSS below 1.40 GB. Native arm64, four Torch threads
and zero data workers. Atomic checkpoints every 40 steps
retain optimizer and sampler state; the same command resumes.
The heartbeat records fold, PID and progress. Forty focused
tests passed, including rotation, translation, recipient swap,
ranking and missing-step accounting. No CREATE job, reserved
confirmation access, Stage5C, SMC, manuscript or deployment
change. [Protocol](registration.json), [freeze](freeze.json),
[summary](summary.json), [attribution](attribution.json).
