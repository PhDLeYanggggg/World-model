# Better Global Cost Prediction Did Not Produce Better Decisions

## Evidence Status

This is a fresh developmental readout of 72 registered source-internal training
runs. See `verification.json` for completed hash, replay and independent
arithmetic checks; documentation alone does not establish those checks passed.
Only the same 12 opened localities are used. Independent model selection,
calibration and confirmation remain not_run and closed. Deployment is unchanged.

## What Worked

Selecting a checkpoint using held recordings from its original training locality
improves the primary transferred signed-score MSE relative to the fixed final
checkpoint: difference -0.706815, nominal 95% locality interval
[-0.970978, -0.456769]. All 12 locality-average differences have the same direction.
This supports source-internal validation for this global prediction-error
criterion on the exposed development pool. It does not certify selected risk.

The comparison is controlled within each run: identical optimization data,
normalization, features, loss, initialization, sampled training queries and
full 2,000-update budget. Transfer labels did not choose checkpoints. All 72
source-internal splits use disjoint recordings; no source had to be dropped.

## What Failed

Same-query count-matched ADE worsens 0.539532% relative to the final policy;
nominal interval [-0.738619%, -0.341516%] when expressed as improvement.
The three seed improvements are -0.660425%, -0.491940% and -0.466233%.
Eleven of twelve locality means worsen; the remaining gain is only +0.006177%.
This is not a reduction in intervention count masquerading as an accuracy gain:
the matched policies have identical per-query counts, though eligible sets differ.

| Matched policy | All ADE gain vs frozen floor | Hard gain | Worst easy ADE degradation | All-risk violations | Easy-risk violations | Undefined all/easy risk |
|---|---:|---:|---:|---:|---:|---:|
| Final checkpoint | +1.04453% | +1.39969% | 25.03774% | 177/216 | 177/216 | 21/21 |
| Validation-selected | +0.52640% | +0.59105% | 21.53520% | 126/216 | 148/216 | 21/21 |

Fewer risk violations are not a safety pass. Undefined views remain undefined,
and selected unknown-label occurrences remain 15,084 versus 11,887. Mean easy
ADE gain is positive (+1.11182% and +2.03985%), yet worst-view easy degradation is
far above 2%. Aggregate improvement must not hide those failures.

Without count matching, final versus selected all gains are +1.62667% versus
+0.56809%, with intervention rates 20.88650% versus 16.77236%. Those policies
change both coverage and composition and are not a pure ranking comparison.

## Failure Taxonomy

1. **Objective-to-decision mismatch:** a lower whole-pool signed-cost MSE did not
   select a better intervention policy. The model-selection criterion is not
   equivalent to selected utility or selected positive-harm risk.
2. **Learning/transfer tension:** 21/72 fits select step 0, and another 20 select
   step 200. This is consistent with limited transferable benefit from later
   fitting on some sources, not proof that all neural learning is useless.
   Step-zero heads are training-prior decoders, not learned representation lift.
3. **Residual conditional risk error:** even after validation selection, 148/216
   matched easy-risk views violate the original selected-reference 2% screen;
   another 21 are undefined. Validation is not an independent calibration set.
4. **Nonuniform easy behavior:** better mean easy ADE coexists with substantial
   worst-view harm. Pooled averages cannot serve as a deployment certificate.
5. **Missing outcome support:** unknown interventions are explicitly counted.
   Future label availability cannot be used to retrospectively restrict actions.
6. **Research exposure:** these are repeated dependent development comparisons.
   Their nominal intervals do not reverse earlier adaptive exposure or establish
   held-out safety, cross-dataset generalization or a new dynamics contribution.

The previous full-source capacity experiment is not a matched counterfactual for
this run: reserving recordings changes the fitting distribution. Its much smaller
worst easy degradation cannot be attributed solely to checkpoint selection.
The admissible primary contrast is final versus validation-selected here.

## Next Controlled Action

Do not extend training or tune thresholds on these transfer readouts. First test
whether a decision-aware source-validation rule can choose among the already
frozen initial, final and MSE-selected heads without sacrificing the original
2% all/easy risk screens. Freeze that rule before reading its transfer outcomes,
retain the MSE-selection comparator, and report fallback/undefined support rather
than calling empty intervention safe. This is a new developmental comparison,
not a post-hoc replacement of this experiment's primary endpoint.

The initial control should reuse verified checkpoints and introduce no new
architecture. If source validation cannot support a nonempty useful policy, that
is a data/conditional-risk support limitation to report, not permission to relax
the risk budget or open independent confirmation for iterative tuning.

Obs8/pred12, stride12 raw frames, image-local detector-silver only. No seconds,
metric, human-gold, true-3D, foundation or physical-safety claim. No Stage5C
execution or SMC.
