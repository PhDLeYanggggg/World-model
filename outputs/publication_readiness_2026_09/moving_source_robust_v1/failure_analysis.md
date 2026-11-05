# Source-Robust Continuation: Failure Analysis

## Result

Six actual neural continuations completed the fixed 1,000-update budget, with
an additional exact 1,000-update refit and 400 real pilot updates. All fitted
metrics, source decompositions and error attributions replay exactly. This
is a completed training experiment, not a NumPy surrogate or a proposed run.

The fixed internal follow-up screen fails **0/3**. No additional nested-C or H
prediction readout is performed. There is no new test score, safety pass or
deployment promotion. The previous 17-view failure remains unchanged.

| Context | Worst-source loss reduction | Joint-cost reduction | False-safe-cost reduction | False-reject-cost reduction |
|---|---:|---:|---:|---:|
| critic0 | +2.1492% | -1.4646% | -5.0465% | -1.4323% |
| critic1 | +6.9365% | -8.2327% | +31.4085% | -9.0615% |
| critic2 | +1.9804% | +1.1602% | -22.4348% | +1.8982% |

Positive reductions mean better. All values above are matched own-fitting
comparisons against the equally trained mean-loss control, not trajectory
improvement percentages or cross-domain generalization results.

## Failure Taxonomy

**Source tradeoff.** The highest-loss sources are 067, 076 and 059. Their
original losses decrease, but losses increase in the other five sources:
048 +6.5935%, 053 +8.9043%, 065 +2.0595%, 060 +4.9186%, 073 +3.7611%.
The observed update favors the worst source at a cost elsewhere. This does
not prove that every group-robust method fails or that domain shift is the
only remaining cause. The fixed mixture, starting models and finite budget
are the scope of this finding.

**Surrogate-versus-decision mismatch.** A lower maximum combined training
objective does not guarantee a lower cost after the unchanged sign-based
admission rule. Risk-MSE reductions are -0.8688%, +5.4689%, -0.1660%.
The three contexts have different loss-component balances, so a universal
"MSE dominates everything" explanation is not supported either.

**Safety-versus-utility tradeoff.** critic1 reduces false-safe cost by31.4085%
but increases false-reject cost by9.0615%; total cost is8.2327% worse. It is
not an unqualified improvement to refuse more useful interventions. critic2
reduces false-reject cost enough to improve the total, while harmful admission
cost increases22.4348%. The predetermined screen correctly rejects both.

**Count-versus-severity mismatch.** critic0 has14 versus15 false-safe fitting
proposals and2589 versus2588 false rejections. One fewer false rejection does
not imply lower missed-benefit cost. The reports retain weighted costs and
counts separately. A zero control cost is NA for percentage reduction, not
a perfect safety score or an identity certificate.

**Limited generalization evidence.** There are only two or three own-B sources
per context, one producer seed, and repeated overlapping windows. All sources
were already part of development. A fixed fitting result cannot establish
generalization, and passing a numerical replay cannot supply an independent
test. No source is removed because its result is unfavorable.

## What Changed And What Did Not

The only treatment change is0.75 L +0.25 max-source L. Both arms retain524
causal input fields,68,356 parameters, exact parent starts, original target
bounds, identical query draws and symmetric optimizer resets. Both encoders
update. No held outcomes choose checkpoints or thresholds. The failed branch
is not rescued by changing the mixture coefficient after viewing its result.

Native obs8/pred12 image-coordinate data and unknown-label bounds remain
unchanged. There is no future endpoint, outcome-availability flag, easy truth
or source-specific threshold in inference. This is intervention-risk learning,
not new trajectory dynamics, JEPA training or a generative world model.

## Next Model Question

Do not repeat proximity-feature additions or tune this source-weight mixture
on held outcomes. The next model-side question is whether the complete-query
admission cost should be represented more directly, instead of expecting a
maximum of separately regressed channel scores to yield a useful safe action
set. First check the existing query-head implementation and prior negative
experiments to avoid duplicating them; then register a matched action-set risk
head experiment using only own-B fitting roles. This is an untested hypothesis,
not a claim that a repair has been found.

The original 2% all/easy positive-harm, signed easy-net, reference and support
rules remain binding. Missing support cannot be repaired merely by fitting
better scores; it remains NA without the required evidence. No identity
exemption, new reserved test, source exclusion, deployment or submission-ready
claim. Original24-group primary remains NA. Stage5C/SMC remain off.
