# Selected Harm, Not Candidate Ranking, Is The Main Risk Error

I kept the forecast bank and original filter fixed and calibrated the cost
estimates on other sources. For each held A source and seed, its 13 donors
receive predictions from cost models that exclude both the held source and
the donor. There are 273 distinct inner ridge fits (91 pairs of exclusions
times three actual forecast seeds), 546 donor readouts, and 42 calibrators.
No neural trajectory model is retrained. The raw held choices and raw
recording scores reproduce exactly. These are exposed development results.

| Calibration | Switches | Nonempty / 102 | Original passes / 102 | Undefined easy | Easy completion >2% | Completion lower vs CV (pp) |
|---|---:|---:|---:|---:|---:|---:|
| None | 6,184 | 92 | 2 | 36 | 64 | +0.07746 [0.02877, 0.13321] |
| Harm only | 694 | 42 | 24 | 70 | 8 | -0.00009 [-0.00045, 0.00015] |
| Reference only | 1,645 | 54 | 14 | 62 | 26 | +0.00837 [-0.00057, 0.02524] |
| Both (primary) | 474 | 32 | 19 | 78 | 5 | +0.00004 [-0.00004, 0.00015] |

The primary correction passes more views, but keeps only 7.7% of the raw
switch count and does not establish positive utility over CV. Its direct
paired completion-lower change versus raw is -0.09197 pp
[-0.14576, -0.04588]. That direct bound is not the difference of two
separate CV-relative bounds. The intervals resample 14 source groups 3,000
times after averaging the three producer seeds. Harm-only yields the most
passes, but it was a diagnostic arm and also has no positive gain interval.
No arm is promoted.

## Which Cost Is Wrong On The Same Actions?

On the original selected paths, I replaced one predicted component at a
time with the corresponding observed/completion component. This uses
future outcomes for attribution only; it does not create any decisions.

| Easy-risk ratio on frozen raw choices | Defined | Over 2% |
|---|---:|---:|
| Predicted harm / predicted reference | 86 | 0 |
| Actual/completion harm / predicted reference | 86 | 57 |
| Predicted harm / actual reference | 66 | 9 |
| Original completion harm / actual reference | 66 | 64 |

Across the dependent selected views, predicted easy numerator mass is
63.06 versus the actual completion mass 4,885.03. The pooled reference is
18,447.51 predicted and 20,806.81 actual, so there is no uniform upward
denominator bias: the major issue is underestimated selected harm, with
local reference errors and absent support still important. These pooled
masses describe the fixed actions; they are not macro-source risk estimates.
Prior evidence already showed 63 of the 64 violations on known labels.

**Decision.** Add a selected-region harm model that separates probability of
positive harm and its magnitude; compare it at matched counts before any
tail correction. The result does not warrant tighter thresholds or using
the closed external data to choose a new model. Independent calibration
and confirmation remain unrun. The 2% and support rules stay fixed.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_nested_action_calibration.py fit
.venv-pytorch/bin/python scripts/run_m3w_nested_action_calibration.py score
.venv-pytorch/bin/python scripts/run_m3w_nested_action_calibration.py replay
.venv-pytorch/bin/python scripts/audit_m3w_selected_action_components.py
```

Registration `09441112`. Native arm64, 4 numerical threads, no worker
processes. Fit/score/replay/attribution exit 0 (40.93/6.84/6.79/5.73 s).
Atomic seed checkpoints; peak fit RSS 2.65 GB. No remote job needed.
No deployment, metric/seconds or physical safety claim, Stage5C or SMC.

[Protocol](protocol.md) | [Results](analysis.json) | [Component audit](selected_component_audit.json)
