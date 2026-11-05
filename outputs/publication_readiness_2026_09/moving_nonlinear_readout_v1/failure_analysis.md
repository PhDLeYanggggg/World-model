# Conditional Readout Failure And Mechanism Analysis

The prespecified compute-allocation condition fails. Do not promote this variant or select a favorable context.

## What This Comparison Can Identify

Both arms retain the exact neural representation, offset, supervision,
query weights, leaf/round budget, unknown-label bounds and decision rules.
Only access to 64 hidden columns differs. This is a standard nonlinear
readout feature ablation, not a comparison of newly trained encoders.
A gain over the affine control alone does not show latent contribution:
the score-only nonlinear control is required. Neither gives a transferred
multicalibration or safety theorem. Tree complexity and effective fit
capacity can still differ when feature dimension changes.

## Result-Specific Diagnosis

The hidden-feature arm improves fitting MSE in all three contexts and
has a positive fitting utility contrast against the score-only arm.
That advantage does not transfer: its held known utility contrast is
-0.075521 pp, with completion bounds [-0.180675, 0.029634] pp.
Those bounds address missing outcomes, not sampling uncertainty. This
is no demonstrated downstream benefit of the frozen latent, not proof
that all neural representations or the underlying data are inadequate.

Both readouts intervene much less than the affine control, losing net
beneficial actions as well as harmful ones. Every registered context
has a negative latent-minus-affine completion lower contrast. The
latent arm retains 13 all-risk and 15 easy-risk violations across 17
recording views; two selected-easy views additionally become undefined.
A lower worst signed easy-net degradation does not pass the separate
positive-harm budget or manufacture reference support.

The action-flow sums below distinguish this from simple over-rejection:
the retained and newly admitted actions still have positive observed
all/easy budget excess despite negative predicted excess. The model
has not learned the admission boundary reliably. Rejected net benefit
cannot justify reopening those actions without a new valid risk model.

## Actual Original Risk Checks

| Policy | Known gain vs floor % | Hard gain % | Switch % | Full risk/support | All violations | Easy violations | Undefined selected easy |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditioned_veto | 10.319488 | 10.208646 | 43.860359 | 1/24 | 19 | 20 | 1 |
| mean_veto | 6.992155 | 7.782569 | 22.115276 | 2/24 | 15 | 15 | 6 |
| quantile90_veto | 0.955191 | 1.092777 | 1.565112 | 3/24 | 4 | 4 | 17 |
| readout_veto | 7.722350 | 8.555668 | 24.195528 | 2/24 | 17 | 18 | 4 |
| score_only_veto | 3.324493 | 3.738445 | 6.589225 | 4/24 | 6 | 5 | 14 |
| latent_score_veto | 5.048838 | 5.042989 | 10.804138 | 4/24 | 10 | 7 | 11 |

The preceding table is fitting-only, not validation. The held table is
in the main report. Lower fitting MSE cannot replace either gate.

## Conditional Risk Error

Pooled available-label sums in native image units, repeated across context
views. They are not independent observations or certification. Unknown
labels retain their bounds in the full policy evaluation, not here.

| Scope | Model | Set | Known | Unknown | Predicted all/easy excess | Actual all/easy excess |
|---|---|---|---:|---:|---|---|
| fit | readout | proposals | 48061 | 1274 | 359549.803319, 171096.718289 | 235500.783506, 45001.300410 |
| fit | readout | admitted | 13616 | 91 | -69350.244655, -66343.177724 | 2701.506722, 2486.750053 |
| fit | score_only | proposals | 48061 | 1274 | 335950.165652, 151129.934483 | 235500.783506, 45001.300410 |
| fit | score_only | admitted | 1923 | 0 | -3592.677968, -731.630530 | -2655.046512, 3.644158 |
| fit | latent_score | proposals | 48061 | 1274 | 326932.185615, 140570.409724 | 235500.783506, 45001.300410 |
| fit | latent_score | admitted | 3828 | 8 | -12593.551462, -4954.935768 | -7753.188644, 87.469858 |
| held_SOURCE | readout | proposals | 78665 | 1444 | 411408.113596, 309121.122754 | 63859.061813, 37718.028649 |
| held_SOURCE | readout | admitted | 35207 | 513 | -252915.118574, -283825.380164 | 17154.654914, 12988.426904 |
| held_SOURCE | score_only | proposals | 78665 | 1444 | 534984.558474, 431454.774385 | 63859.061813, 37718.028649 |
| held_SOURCE | score_only | admitted | 14097 | 101 | -51652.902900, -55974.385552 | 11027.709535, 7972.687352 |
| held_SOURCE | latent_score | proposals | 78665 | 1444 | 687407.938843, 567974.198438 | 63859.061813, 37718.028649 |
| held_SOURCE | latent_score | admitted | 14708 | 158 | -54733.335281, -58708.662801 | 10812.020268, 8081.255554 |

## Which Proposals Changed?

| Scope | Model | Flow vs affine | Known | Unknown | Net benefit | Positive harm | All excess | Easy excess |
|---|---|---|---:|---:|---:|---:|---:|---:|
| fit | score_only | retained | 1731 | 0 | 65557.419710 | 490.000020 | -2445.675736 | 9.119948 |
| fit | score_only | newly_rejected | 11885 | 91 | 179602.296095 | 21835.287023 | 5147.182458 | 2477.630105 |
| fit | score_only | newly_admitted | 192 | 0 | 5593.986422 | 36.514843 | -209.370776 | -5.475790 |
| fit | score_only | still_rejected | 34253 | 1183 | 147534.367640 | 271337.063396 | 233008.647559 | 42520.026147 |
| fit | latent_score | retained | 2939 | 6 | 108636.006409 | 1656.808400 | -4563.556125 | 97.028064 |
| fit | latent_score | newly_rejected | 10677 | 85 | 136523.709396 | 20668.478643 | 7265.062847 | 2389.721989 |
| fit | latent_score | newly_admitted | 889 | 2 | 74279.315408 | 263.927151 | -3189.632519 | -9.558206 |
| fit | latent_score | still_rejected | 33556 | 1181 | 78849.038654 | 271109.651088 | 235988.909302 | 42524.108563 |
| held_SOURCE | score_only | retained | 13632 | 98 | 78831.460617 | 17398.730234 | 10388.775809 | 7475.144521 |
| held_SOURCE | score_only | newly_rejected | 21575 | 415 | 146702.428178 | 20223.695243 | 6765.879105 | 5513.282384 |
| held_SOURCE | score_only | newly_admitted | 465 | 3 | 1084.526506 | 834.761009 | 638.933726 | 497.542831 |
| held_SOURCE | score_only | still_rejected | 42993 | 928 | 255555.001566 | 68680.703452 | 46065.473174 | 24232.058914 |
| held_SOURCE | latent_score | retained | 14094 | 136 | 75875.127648 | 17798.469363 | 10510.082135 | 7743.748922 |
| held_SOURCE | latent_score | newly_rejected | 21113 | 377 | 149658.761147 | 19823.956113 | 6644.572779 | 5244.677982 |
| held_SOURCE | latent_score | newly_admitted | 614 | 22 | 2381.911710 | 673.023663 | 301.938134 | 337.506632 |
| held_SOURCE | latent_score | still_rejected | 42844 | 909 | 254257.616362 | 68842.440798 | 46402.468766 | 24392.095113 |

## Every Registered Context

| Context | Latent minus affine completion lower pp | Affine all/easy violations | Latent all/easy violations | Expansion condition |
|---|---:|---|---|---|
| eqmotion_fold1_seed17_critic0 | -1.555480 | 9/9 | 9/9 | False |
| eqmotion_fold0_seed17_critic1 | -10.859428 | 2/4 | 2/4 | False |
| eqmotion_fold0_seed17_critic2 | -0.657500 | 2/2 | 2/2 | False |

## Limits

New fitting error reduction is not independent generalization or a new
dynamics contribution. Inspect retained harmful actions and discarded
beneficial actions together; fewer interventions alone are not success.
The same source is used for the fixed encoder and its readout fitting,
so fitting diagnostics are resubstitution, not calibration evidence.
Held SOURCE is already exposed development, not fresh confirmation.

Any absence of easy-reference mass remains NA under the original rule.
Changing a nonlinear score cannot mathematically create missing labels
or reference support. No favorable source, seed or checkpoint is selected
after this readout. Any next model needs a new frozen comparison.

No metric, seconds, human-gold, physical-safety, true3D, foundation or
submission-ready claim. No new H; Stage5C and SMC remain off.
