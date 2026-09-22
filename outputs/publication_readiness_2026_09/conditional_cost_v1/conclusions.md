# Fixed Decision-Region Weighting: Accuracy Gain, Protection Failure

## Evidence Status

Completed on 2026-09-22 after registration commit `623e02b0`. Twelve real Torch
cost heads are `fresh_run`; forecasts and 36 equal-budget reference heads are
`cached_verified`. The four SDD sites remain research-design exposed. Independent
calibration, final confirmation and deployment are `not_run`, not successful.

This experiment uses eight observed and twelve predicted annotation steps,
SDD stride12, annotation pixels. Raw-frame t50 is a separate supplement. This is
not metric, seconds-level, true3D, foundation-model or population-safety evidence.
Stage5C and SMC remain off.

## What Changed

Only the fitting loss weight changed: fourfold emphasis on the frozen preceding
head's strict-selected fitting rows, normalized under the same training sampler.
Both benefit and harm errors receive the same weight. Architecture, 356 causal
features, initialization, forecasts, targets, row draws and inference thresholds
are unchanged. Each of 12 heads has 45,954 parameters and 12,000 updates. The
100-update pilot was resumed, not counted as an additional experiment.

New compute: 144,000 updates, 36,864,000 training draws, 133.80 recorded head-fit
seconds. This excludes prior predictor fitting, I/O and verification. It is a
small downstream-head experiment, not a fresh full world-model training run.

## Fixed Primary Result

| Metric | Previous strict head | Region-weighted strict head |
|---|---:|---:|
| Equal-site ADE improvement over causal CV | 3.72892% | 4.09764% |
| FDE improvement | 4.03089% | 4.41091% |
| Hard-subset improvement | 3.74386% | 4.25956% |
| Positive-easy degradation | -0.81395% | -0.57490% |
| Complete exact-zero-CV harmed query/seed instances | 0 | 0 |
| Selected query/seed instances | 28,565 | 29,668 |
| Selected instances with no ADE label | 398 | 423 |
| Selected instances with incomplete futures | 3,917 | 4,194 |

Negative easy degradation means improvement. Unknown-label instances are included
in incomplete counts; these columns must not be added. Counts across seeds are
not unique independent observations.

The fixed primary ADE-gain contrast is **+0.36871 percentage points**, paired
3,000-site-bootstrap CI **[+0.17316, +0.60302]**. The new gain relative to CV has
CI **[2.62300%, 5.99397%]**. All seeds improve CV, and aggregate easy preservation
passes. Nevertheless, the preregistered joint development gate **fails** because
every scene/seed must also preserve easy cases within 2%.

| Scene | Seed17 easy degradation | Seed29 | Seed43 | Seed-average |
|---|---:|---:|---:|---:|
| coupa | -7.69884% | -6.73178% | -6.75038% | -7.06033% |
| deathCircle | 3.28309% | 2.84343% | 2.67344% | 2.93332% |
| gates | 3.85432% | 1.37039% | 1.22712% | 2.15061% |
| hyang | 0.12794% | -0.36143% | -0.73609% | -0.32319% |

The new head does not repair deathCircle's protection failure and adds a gates
seed17 failure. Aggregate gains must not conceal either. No new model is deployed.

## Secondary Controls

| Policy | ADE gain | FDE gain | Hard gain | Easy degradation | Exact-zero harms |
|---|---:|---:|---:|---:|---:|
| Net benefit minus harm | 11.95078% | 12.78719% | 14.22402% | 21.68172% | 21 |
| Fixed strict rule | 4.09764% | 4.41091% | 4.25956% | -0.57490% | 0 |
| Frozen matched intervention count | 5.50627% | 6.01973% | 8.59967% | 6.12999% | 0 |

At matched counts the new head is **-0.04374 points** below the old intermediate
head, CI [-0.07376, -0.01372]. Therefore the strict-rule gain does not establish
better ranking at equal intervention volume. The corresponding strict contrasts
against native and fraction controls are +1.03045 [0.17847,2.33747] and +2.43997
[0.80544,4.95236] points, respectively. They are secondary, unadjusted development
comparisons, not replacements for the failed combined primary gate.

## Failure Diagnosis

Same-population costs use complete futures only. The 72 records compare old and
new heads on all rows, the fixed old selection region and the new selection
region, separately for fitting and outer-held sources.

| Population/region | New harm underestimated, of12 views | New joint cost MSE lower, of12 |
|---|---:|---:|
| Fitting/all | 1 | 3 |
| Fitting/old selected region | 5 | 11 |
| Fitting/new selected region | 11 | 9 |
| Held/all | 5 | 6 |
| Held/old selected region | 12 | 9 |
| Held/new selected region | 12 | 7 |

The training emphasis usually improves cost MSE on the region it targets. It
does not keep the head reliable after the head selects a new region. On newly
selected deathCircle held rows, predicted mean harm is 0.869/0.778/0.696 pixels
versus observed 2.949/2.210/3.454 across the three seeds. These are whole selected
complete populations, not easy-only numbers. This supports selection-conditioned
optimism as an unresolved failure, not a proof of a particular causal mechanism.

Lower bounds for complete-grid gain remain negative for gates in seeds17/29
when unobserved outcomes are retained. No observed-zero-harm result can establish
safety for missing futures. Continuous costs are not calibrated probabilities.

## Verification and Next Step

The new 12 checkpoints replay all 527,268 score rows. Separate label, weight,
policy and metric arithmetic checks the same recorded result; see the receipts
and [execution notes](execution_notes.md). This is same-agent verification, not
independent research replication. Thirty-nine unchanged scoped tests passed
before registration; the nonhermetic legacy full suite was not rerun.

The current contribution is a modest developmental accuracy gain plus a failed
protection repair. It does not close the independent calibration or confirmation
gap and does not establish CVPR readiness. Do not sweep the multiplier or held
thresholds after this result. The next useful question is whether costs evaluated
on genuinely out-of-fitting selection regions can support reliable intervention,
without simply abstaining or hiding per-scene harm. Audit the necessary separate
fitting/calibration roles before another prospectively fixed experiment; retain
the existing closed original roles and request a decision if new roles are needed.

Full analysis SHA256:
`52bef548c000a24c1c99b0e33081371bd5caf78be42ea60525383619e285788e`.
