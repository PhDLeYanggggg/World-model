# Frozen Harm-Error Diagnosis

## Evidence

Fresh frozen-model inference and diagnostic calculation completed for all72
heads, all12 exposed development localities and all three head seeds. Models
and their training inputs are cached_verified by hashes; no new model was
trained in this diagnostic. No independent selection/calibration/confirmation
role was opened. No deployment or scientific gate was promoted.

The first full attempt was interrupted after48 heads. The resumed run replayed
those48 exactly and completed all72, exit0,245.54s, peak10.223GB RSS. It read
434.25MB of immutable owned CREATE checkpoints into memory, wrote no local
numerical cache and launched no HPC scientific job.11,592 scalar checks and
9,209 independent scalar/aggregate/bootstrap checks pass.

There are596,988 repeated validation row occurrences across model views,
including14,076 unknown occurrences. These are not independent sample counts.
All uncertainty below is nominal equal-locality paired3000 bootstrap over12
already exposed development localities, not a confirmatory generalization CI.

## What Explains the Failed Cost Accuracy?

| Quantity | Change / contribution | Nominal95% locality CI |
|---|---:|---:|
| Final normalized signed-score MSE | +0.136161 | [0.039763,0.257608] |
| Easy-harm moment contribution | +0.123727 | [0.034404,0.247105] |
| Total-harm moment contribution | +0.012348 | [-0.000245,0.028463] |
| Benefit contribution through projection | +0.00008586 | [-0.00001478,0.00024727] |
| Reference / easy-reference contribution | 0 / 0 | [0,0] |
| Before projection | +5.029890 | [0.213045,13.981435] |
| Positive-arm projection effect | -4.901342 | [-13.770706,-0.170822] |
| Original-arm projection effect | -0.007613 | [-0.019514,-0.000626] |
| Additive control vs original | -0.032146 | [-0.064672,-0.007822] |

Easy harm accounts for90.87% of the algebraic final error increase. Projection
substantially limits the much larger raw error; removing it is not supported.
This is an exact quadratic decomposition, not causal ablation or evidence that
one feature alone caused the problem.

The original-selected cohort contributes only+0.000001991, whereas the
original-unselected cohort contributes+0.136159. On the positive-selected set,
the same fixed prediction comparison improves MSE by0.000007310; on its
complement it worsens by0.136168. Thus aggregate cost error and realized
selected-policy risk must not be conflated. Tiny selected prediction error
does not cancel the two known-label and nine additional unknown-completion
violations from the prior training result.

## Tail and Support Evidence

The easy-harm raw prediction ratio>4 slice contributes+0.144035
[0.046234,0.267215], more than the net increase because other slices improve.
The easy-harm overprediction slice contributes+0.145058, while under/equal
prediction contributes-0.008897. These label-based slices are offline diagnosis
only; no such flag is available to the deployment policy.

The slice with at least half of trees having TRAIN effective easy-harm support
below5 contributes+0.123986 [0.027143,0.249040]. Low mean TRAIN normalized
easy-harm magnitude<=0.1 contributes+0.089529. These strata overlap; their
contributions must not be added or treated as independent causal effects.
Effective support measures weighted TRAIN rows, not independent scenes.

This supports a hypothesis of rare-event, low-magnitude tail amplification
under conditional relative-deviance fitting. It does not prove a squared-cost
loss will solve selection risk. The next controlled experiment keeps positive
mean-preserving form, features, forest routing, source partitions, masks and
2% budget fixed, and changes the fitted objective to the original signed-score
quadratic surrogate. No post-hoc cutoffs or label-conditioned sample removal.

## Numerical Amendment and Limits

The v1 diagnostic stopped at a5.32251621e-9 score mismatch on head4. All real
targets were float64: the actual cause was multiplying scalar scale by float32
RMS before division. The maximum real-head discrepancy is5.6932e-8. The old
reader divided separately. v2 reproduces that order without widening tolerance.
The float32 target-transform issue was demonstrated only in a synthetic
regression; the earlier protocol's suggestion it affected these real labels
is corrected here. v1 code, registration, pilot and three reports are retained.

The overall scientific goal remains incomplete. Labels are detector-silver,
coordinates image-local, observation8/prediction12 at rawstride12. No metric,
seconds, human-gold, physical-safety, true3D, foundation or submission-ready
claim. Stage5C and SMC remain off.
