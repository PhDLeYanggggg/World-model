# What the Complete v6 Study Establishes

Date: 2026-09-17. Result source: fresh training/evaluation, with hash-verified
row caches for the post-run error analysis. Development only, not confirmation.

## Question and Budget

Can a stronger public forecasting core, trained at a complete matched budget,
provide useful candidates for baseline-relative joint intervention on the
eight-observed/twelve-predicted native-step task?

The comparison completes both registered families and all seeds 17/29/43.
There are 24 forecasting fits at 10,000 updates, six neural benefit/harm heads
at 1,000 updates, and six OOF ridge controls. EqMotion uses its version-pinned
author core with one fixed head, not published best-of-20 evaluation. The local
Transformer uses the same fit windows, observed neighbor support, loss and
update/sample budget. Parameter counts, hardware and FLOPs are not equated.

## Results That Failed

| Seed | Transformer primary gain vs CV | EqMotion-K1 primary gain vs CV | Development choices |
| --- | ---: | ---: | --- |
| 17 | -5.736% | -14.119% | Both CV |
| 29 | -7.686% | -8.473% | Both CV |
| 43 | -6.700% | -14.081% | Both CV |

All 120 main arm comparisons remain in the family reports. Neither family
passes the prescribed improvement/easy-preservation selection. The diagnostic
oracle can improve the frozen primary error by only 0.455--0.604% for EqMotion
and 0.506--0.579% for Transformer. This oracle sees future labels; it is not a
learned predictor. Its bound rules out obtaining 5% improvement simply by
retuning a switch between these unchanged candidates and CV.

The two native-coordinate recordings tell different stories. On Students03,
EqMotion beats the development-best causal alternative by 2.103%, 2.318% and
0.940%. On Students01 it loses by 4.966%, 5.691% and 2.126%. Both recordings
belong to one physical University site. This is a conditional positive signal,
not success on the frozen primary metric or stable cross-scene generalization.
Transformer does not beat that stronger native-coordinate baseline on either
recording in any seed.

## What Was Repaired

1. The previously observed numerical overflow no longer occurs within the full
   v6 registered budgets. Inputs are conditioned using only observed context;
   outputs are restored before the original loss and evaluation. This is a
   demonstrated execution repair, not a prediction improvement.
2. Students01 continuous identities, short tracks and tails are retained when
   past-supported, rather than using fixed twenty-point packaging for context.
   Fit rows are unchanged. This repairs a source-population issue for the
   full-scene task, not historical test exposure or all upstream interpolation.
3. Every seed/fold completes and remains reported. Twelve fits per family are
   not twelve independent datasets; no scene CI is manufactured from them.

## Error Mechanism: Evidence and Hypothesis

Exactly 3,082 of 28,324 complete ADE queries (10.881%) have the already-frozen
normalization scale at its numerical floor of 0.001 dataset-local units.
They account for 88.88--91.58% of Transformer positive harm and 93.08--96.97%
of EqMotion positive harm. The signed bin contributions reconstruct the
original primary excess error. Easy baseline normalized ADE is 0.00323052,
so large relative degradations must be read alongside absolute excess.

This locates error mass; it does not prove that the scale causes the network's
mistakes. Errors remain net positive above the floor as well. We do not delete
stationary rows, change the metric, or choose a seed using this diagnosis.
Plausible mechanisms for a prospective experiment include a learned drift on
nearly stationary histories, weak identifiability of future stop/start events
from eight positions, and conditioning that mixes a very small ego scale with
much larger neighbor geometry. These remain hypotheses, not established causes.

## Implications for the Publication Route

The immediate target stays one rigorous mechanism paper, not multiple lightly
different architecture papers. This study does not yet justify that submission:
the intervention advantage is unproved, the primary candidate headroom is low,
and independent confirmation/calibration sites remain unavailable.

The shortest useful next experiment is a separately frozen, fit-designed
motion-state-aware predictor with unchanged evaluation, followed by the same
three-seed comparison and cost-sensitive deferral control. It should test
near-stationary preservation explicitly, not add another threshold sweep to
these failed forecasts. New visual context is worthwhile only if its source
geometry, timing and train-only provenance are first verified and its retrained
ablation has measurable benefit. A positive result on this development site
would still require new independent sites before a generalization claim.

A later journal extension needs a substantively distinct advance, such as
verified visual context and independent multi-domain evaluation, not another
stage number or a larger table of the same exposed windows.

## Unchanged Limits

SDD is not evaluated in this v6 comparison. Native annotation steps are not
verified seconds; dataset-local coordinates are not verified meters. The
source audit identifies unresolved ETH clock mapping and a Students03 matrix
version mismatch. No true-3D, foundation, calibrated physical-safety, deployment,
submission-ready or acceptance claim follows. Stage5C and SMC remain disabled.
