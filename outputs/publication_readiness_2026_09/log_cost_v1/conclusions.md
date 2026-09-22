# Log-Loss Repair: Small Accuracy Change, Protection Still Fails

## Material Passport

2026-09-22. Twelve fresh Torch risk-head fits and a fixed development readout,
followed by checkpoint replay, separate arithmetic verification and a fresh
post-readout turnover diagnosis. Existing forecasters, inputs and reference heads
are cached_verified by their bound hashes. Independent calibration, independent
confirmation and a new external experiment are not_run. All four physical sites
have been used for development. This is not a deployable or submission-ready result.

The [registration](registration.md) preceded training in commit `173b6ce3`.
It fixes the loss as the only changed training factor, with no threshold,
learning-rate, reviewer or seed selection after readout. The known logarithmic
score is not claimed as a new method. Stage5C and SMC remain disabled.

## What Was Actually Trained

Four source-excluded sites (coupa, deathCircle, gates, hyang), seeds 17/29/43:
12 small cost heads, each 45,954 parameters, 12,000 updates and batch size 256.
Total: **144,000 updates and 36,864,000 sampled draws**; zero unknown-label rows
sampled. Recorded head-fit time is **146.91 seconds**, including checkpoint writes
but excluding feature preparation, source hashing, evaluation and earlier forecast
training. This is not twelve newly trained full world models or an HPC experiment.

The loss predicts the same bounded expected benefit/harm costs as the square-loss
control. It changes their supervision to distance-weighted compositional log loss.
Architecture, initialization, complete-label rows, train-only normalization, frozen
4x fitting emphasis, sample draws, optimizer settings and inference remain fixed.
The fractions are not calibrated failure-event probabilities. A better gradient
near zero predicted harm does not establish better out-of-scene risk control.

## Fixed Primary Result

Primary task: 8 observed / 12 predicted annotation steps, stride 12, SDD annotation
pixels. Relative ADE gains are averaged equally across physical sites after seed
error averaging. Seeds are not ensembled forecasts or additional independent sites.
Raw-frame t+50 is a separate supplement and was not evaluated in this experiment.

| Strict policy result | Value |
| --- | ---: |
| ADE improvement over causal constant velocity | 4.18873% |
| Scene-bootstrap CI for improvement over CV | [2.78841%, 6.10667%] |
| Prior frozen-region square-loss ADE improvement | 4.09764% |
| Primary paired difference | +0.09109 percentage points |
| Primary 3,000-scene-resample CI | [-0.00454, +0.21899] percentage points |
| FDE improvement over CV | 4.50140% |
| Hard-subset ADE improvement | 4.51531% |
| Aggregate positive-easy degradation | -0.37375% (improvement) |
| Worst scene/seed positive-easy degradation | 4.10386% (deathCircle seed17) |

The primary superiority interval includes zero. More importantly, protection
fails within scenes even though the aggregate easy metric improves. Every condition
in the registered joint gate is required; **the joint gate fails**.

| Scene | ADE gain over CV | Positive-easy degradation, seed-averaged |
| --- | ---: | ---: |
| coupa | 4.09014% | -6.71643% |
| deathCircle | 7.08796% | 3.38121% |
| gates | 2.41401% | 2.19213% |
| hyang | 3.16281% | -0.35191% |

deathCircle fails in all three seeds (4.10386%, 2.91706%, 3.12271%). gates seed17
also fails (3.46559%). The 2% ceiling was not relaxed. A positive average or seed
does not override these failures.

## Other Fixed Policies, Not Replacement Winners

| Policy | ADE gain | Hard gain | Easy degradation | Harmed complete exact-zero-CV outcomes |
| --- | ---: | ---: | ---: | ---: |
| Strict, primary | 4.18873% | 4.51531% | -0.37375% | 0 |
| Net gain only, with past-stop veto | 11.97048% | 14.31568% | 21.98927% | 21 |
| Pre-existing matched-count ranking | 5.49053% | 8.69040% | 6.23477% | 0 |

At the matched intervention count, the new head differs from the frozen-region
square-loss control by **-0.01574 pp**, CI [-0.06636, +0.03487]. Neither higher
unprotected gain nor a comparison against a weaker reference establishes success.
All three policies were fixed before readout; none is promoted after seeing scores.

## What the Loss Changed, and What It Did Not

On the old held-source selected regions, the new head increases predicted mean
harm in all 12 views. Nevertheless, it still underestimates realized mean harm in
all 12. On its own newly selected held-source regions, underprediction also
persists in **12/12** views. Lower complete-row MSE appears in only 5/12 of these
new regions, versus 10/12 corresponding fitting regions.

This supports a narrower conclusion than "log loss failed everywhere": stronger
local corrective gradients and improvements on familiar selected cases were
insufficient for reliable conditional generalization. It does not identify whether
missing causal information, source shift, finite model capacity or fitting dynamics
is the unique cause. Square loss is not an improper objective by this evidence.

### Fresh Frozen-Decision Turnover Diagnosis

The [turnover audit](turnover_diagnosis.json) partitions old and new strict decisions
without training or changing a decision. Counts sum over repeated seeds and are
**not independent samples**. Conditional benefit/harm uses complete future labels
only; the exact gain decomposition retains the primary supported-ADE population.

| Decision group | Rows | Complete labels | Beneficial complete | Harmful complete | Entirely unknown labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained by both | 22,908 | 20,014 | 17,565 | 2,449 | 289 |
| Dropped by log loss | 6,760 | 5,460 | 3,872 | 1,588 | 134 |
| Added by log loss | 6,054 | 4,835 | 3,532 | 1,303 | 157 |

The new head underestimates mean harm in both the retained and added groups in
all 12 views. The unchanged retained decisions therefore remain part of the
problem; it is not solely a turnover or stale-training-region effect.

For each site and seed, all contributions use the same full easy-subset reference
error denominator. Thus new gain = retained contribution + added contribution;
old gain = retained contribution + dropped contribution. Negative is degradation.
These are algebraic contributions, not performance percentages within each group.

| Easy-subset contribution, seed mean (pp) | Retained | Dropped | Added | Old gain | New gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| coupa | +6.73843 | +0.32191 | -0.02199 | +7.06033 | +6.71643 |
| deathCircle | -2.41647 | -0.51685 | -0.96474 | -2.93332 | -3.38121 |
| gates | -1.30956 | -0.84105 | -0.88257 | -2.15061 | -2.19213 |
| hyang | +0.52574 | -0.20254 | -0.17383 | +0.32319 | +0.35191 |

In deathCircle, the common retained decisions already contribute **2.41647 pp**
of easy degradation. Newly added decisions contribute another **0.96474 pp**,
while dropped decisions remove 0.51685 pp of old harm. Merely blaming the added
decisions cannot explain away the pre-existing failure. This is retrospective
diagnosis, not permission to deploy a hand-picked intersection or scene rule.

## Label Support and Statistical Limits

- 175,756 past-eligible queries, 33 recordings, four physical sites; overlapping
  windows and three repeated seeds are not independent statistical units.
- 172,957 queries support ADE; 144,010 support FDE; 143,918 have all 12 labels.
  The strict complete-only ADE gain is 4.43223%, a different support population.
- Across seeds, strict selects 28,962 decisions, including 4,113 incomplete and
  446 entirely unknown outcomes. Unknown outcomes are not counted as safe.
- Full-grid gain bounds remain negative at the lower bound in gates for all
  three seeds. Complete exact-zero-CV harm count zero is not a full-grid guarantee.
- The 3,000 resamples are physical-site resamples from only four design-exposed
  sites. This is a development interval, not independent confirmation or a
  calibrated risk certificate. The matched count control is not a new test split.
- Offline annotations are the input contract. Past numerical inputs and clean
  teacher exclusions do not prove sensor-time annotation provenance.

## Decision and Priority Gaps

**Do not deploy this head. Do not upgrade the contribution or submission claim.**
The previous historical deployment artifacts are unchanged, not re-certified here.

1. Independent scene-level calibration and confirmation support remains the
   highest-priority evidence gap. The pending author choice and acquisition
   approval must be resolved before assigning new data roles. Source development
   cannot be renamed to supply independence.
2. Conditional-risk repair must address persistent harmful retained decisions,
   not only added switches, aggregate MSE or an adjustable risk threshold. A new
   fitting hypothesis needs a fixed comparison before further outcomes are read.
3. The original joint-versus-independent intervention contribution remains
   unproved. This loss test adds neither joint interaction evidence nor external
   generalization, and it does not justify scaling training on CREATE.

The [existing manuscript](../evidence_manuscript_v1/manuscript.md) remains a pinned
earlier evidence snapshot. This negative follow-up is linked separately rather
than silently replacing its comparisons. Reproduction and verification scope are
in [execution notes](execution_notes.md). No metric, seconds-level, true-3D,
foundation, physical-safety, Stage5C or SMC claim is made.
