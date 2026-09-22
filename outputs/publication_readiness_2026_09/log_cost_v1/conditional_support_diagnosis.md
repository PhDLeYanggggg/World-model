# Retained Harm and the Missing Prefix-Cost Target

## Material Passport

2026-09-22. Fresh post-readout diagnosis and a verified label-interface repair,
using cached_verified forecasts, risk decisions and source bindings. No new model
fit, threshold choice, independent calibration, closed-role evaluation or deployment
was performed. All four physical sites remain development-exposed. The scalar
log-loss experiment's failed primary/protection gates are unchanged.

This follows the [frozen turnover audit](conclusions.md): retained decisions,
not only newly added switches, contribute to failed easy preservation. The new
question is whether this reflects a few repeated trajectories, random seeds,
training sampling, or a mismatch between supervised and evaluated future support.

## 1. Harm Is Concentrated, but Not Only a Single Seed

The table concerns positive-easy rows selected by both scalar square-loss and
log-loss policies, with supported ADE. Gross harm is max(neural ADE - CV ADE, 0),
not net harm after benefits. Native error sums are never pooled across scenes.
Repeated seeds are first aligned by the same recording/agent/query key.

| Site | Unique query rows harmed in any seed | Rows harmed in all three seeds | Gross harm from all-three rows | Largest track share | Largest recording share |
| --- | ---: | ---: | ---: | ---: | ---: |
| coupa | 72 | 13 | 49.97% | 12.96% | 43.29% |
| deathCircle | 109 | 23 | 76.05% | 38.38% | 48.63% |
| gates | 50 | 6 | 59.09% | 28.20% | 73.12% |
| hyang | 243 | 48 | 45.28% | 9.11% | 19.66% |

In deathCircle, `video3:119` and `video0:659` together account for over half of
retained-easy gross harm. In gates, `video3:367` and `video3:368` account for
55.07%; video3 contributes 73.12%. These are failure-localization clues, not
permission to remove tracks, tune per-video rules or relabel a test set.

The two failed sites are not explained exclusively by an unlucky seed. Equally,
three seeds do not create three independent datasets. Recording-qualified track
IDs do not establish distinct people across recordings. Connected overlapping
window supports are reported descriptively, not counted as independent events.

Training sampling was checked separately. Within a fitting site, the largest
track has 3.94% of expected draws in coupa, 1.08% in deathCircle, 2.04% in gates
and 1.31% in hyang. Actual draws follow these intended weights. Long tracks have
more mass, but no sampler bug or unique sampling-based cause is established.
These checks do not justify changing the evaluation weighting.

## 2. Most Harm in the Failed Sites Has Only a Short Future Label Prefix

| Site | Gross harm from incomplete future labels | From only one future point | From one to three future points |
| --- | ---: | ---: | ---: |
| coupa | 7.98% | 0.80% | 1.54% |
| deathCircle | **76.80%** | **46.44%** | **70.87%** |
| gates | **74.70%** | **57.54%** | **63.66%** |
| hyang | 31.62% | 19.54% | 25.25% |

These fractions use retained-easy gross harm aggregated over the three seeds
within a site. They are not fractions of all errors or causal effect estimates.
All 768 repeated-seed retained-easy rows with partial support have contiguous
prefix masks. Across all retained subsets, 34 of 2,605 partially supported
decisions have gapped masks: prefix-only protection would not cover arbitrary
missing-point patterns. Entirely unknown outcomes remain explicitly unknown.

The existing cost head was trained only where all 12 future labels were present,
using one scalar benefit/harm pair derived from full-trajectory ADE. The primary
readout, correctly retained, uses available future labels for supported ADE.
Therefore a small full-horizon cost is not necessarily a small cost on an
observed prefix. This is a concrete target-support mismatch; its repair is not
yet a proven remedy for the entire scene-generalization problem.
The earlier 12/12 selected-region underprediction was measured on complete
labels and remains a separate failure. Partial support cannot explain it away.

Do not discard short-support rows, fill missing outcomes with zero, put future
mask length into inference features, or call complete-only results the primary
score. The original support-aware readout and full-grid bounds stay unchanged.

## 3. The Contradiction Already Exists in the Training Supervision

For complete fitting trajectories, we independently reconstructed prefix costs
for every length k=1,...,12 using the existing nested, source-excluded forecasts.
Among trajectories with no full-horizon harm, **44.94% to 63.15%** in the individual
fitting-scene/view groups have harm on at least one shorter prefix.

Across overlapping fitting views and seeds there are 722,680 full-horizon
nonharmful cases; 368,206 have some prefix harm. These counts are repeated-view
instances, not unique independent examples. No significance test is attached.

The scalar loss assigns zero harm to all such full-horizon cases, even though a
prefix loss would be positive. A perfect scalar fit would not recover an entire
prefix-risk profile. This supplies a more specific next training hypothesis than
another scalar loss or threshold adjustment. It does not imply that prefix
supervision will generalize, preserve useful interventions or satisfy the gate.

## 4. Implemented Repair: A Separate Label-Only Prefix Interface

`src/world_model/m3w_prefix_cost_targets.py` now provides two separate functions:

- `causal_prefix_disagreement`: accepts only the two forecasts and the past-derived
  coordinate scale. No target or future availability argument exists.
- `supervised_prefix_costs`: emits the benefit/harm label pair for each prefix,
  plus a label-availability mask. A gap invalidates that and all later prefixes;
  unavailable labels are NaN, never zero. This function is only for loss/evaluation.

For prefix k, let g_k = ADE_CV(1:k) - ADE_neural(1:k). Labels are
`(max(g_k,0), max(-g_k,0))`; disagreement D_k is the mean distance between the
two predictions through k. Triangle inequality gives benefit_k + harm_k <= D_k.
The k=12 label agrees with the already frozen scalar supervision. Numerical
coordinate scale converts back to annotation pixels; it is not meter calibration.

The interface is implemented and real-array verified, **not yet connected to a
newly trained risk head**. No prefix policy result or accuracy improvement is claimed.
The next controlled fitting experiment should keep the admitted rows, sources,
forecasts, preprocessing, seed/budget and existing risk tolerance fixed, compare
scalar versus prefix supervision, and separate a terminal-only rule from a fixed
all-prefix rule and an equal-count control. Register it before fitting/readout.
Future label length must not decide which prefix to use at inference.

This remains supervised risk estimation, not conformal calibration. The primary
ICLR paper's Section 1.1 and Theorem 1 require exchangeable loss functions and a
bounded monotone calibration construction; Section 2.3 explains why monotonicity
cannot simply be assumed. Our overlapping development windows and learned cost
profiles do not establish those conditions. No risk certificate follows from
adding twelve heads. Reading scope: those sections, not a full-paper attestation.
[Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).

The confidence-sequence literature was screened at abstract scope only; a
finite twelve-prefix prediction audit is not automatically anytime-valid
statistical inference. No such theorem or novelty claim is made here.
[Howard et al., author preprint](https://arxiv.org/abs/1810.08240).

## 5. Verification and Reproduction

Thirty new scoped tests passed: 12 concentration cases, 9 horizon-support cases,
and 9 prefix-target cases. They cover partition-independent aggregation, repeated
track/frame rejection, overlap connectivity, missing-label handling, exact-zero
cost, prefix/full sign reversal, terminal scalar equivalence and future-label
perturbation invariance of the causal disagreement interface.

The real-array verifier checked 24 fitting/held views, 288 separate prefix
reductions and 408 sampled queries under perturbation of available future labels.
It covers 1,581,804 fitting
row instances and 527,268 held-source instances, with 17,078,382 and 5,692,794
available prefix labels respectively. These are correlated reused views, not a
new larger dataset. No target cache or large materialization was exported.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_harm_concentration.py
.venv-pytorch/bin/python scripts/audit_m3w_horizon_cost_support.py
.venv-pytorch/bin/python scripts/verify_m3w_prefix_cost_targets.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_harm_concentration.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_horizon_cost_support.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_prefix_cost_targets.py
```

All diagnostic outputs bind the unchanged main analysis and source hashes. The
separate prefix calculation shares source arrays but reconstructs each prefix
directly rather than using the builder's cumulative-sum implementation. It is
same-agent engineering verification, not independent research confirmation.
The existing full-suite limitations remain; no full-suite pass is claimed.
Both diagnosis commands and the real target verifier were rerun; their immutable
outputs matched exactly. All required processes finished with exit 0.

### Registration Erratum, Not a Feature Change

The frozen registration text says 355 causal features. The actual bounded-cost
code and all twelve saved preprocessors consistently use **356 = 355 base cost
features + log(1 + forecast disagreement)**. The 45,954-parameter architecture
matches 356 inputs. The registration stays immutable; this text corrects its
description. No feature, checkpoint, decision, loss or metric was changed.

## 6. What Remains Unproved

The most actionable next method test is prefix-risk learning under the unchanged
evaluation, not deleting incomplete outcomes or sweeping the scalar threshold.
Independent scene-level calibration/confirmation remains a separate unresolved
requirement. Existing role/acquisition decisions are still pending, and no original
closed role was opened. Joint-scene contribution, external confirmation and a
submission-ready main method remain incomplete. Stage5C and SMC stay off.
