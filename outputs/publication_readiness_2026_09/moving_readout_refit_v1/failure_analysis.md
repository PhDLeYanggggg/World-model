# Optimization Versus Admission Error

The prespecified allocation condition is not met. Do not promote this output-only repair.

## Controlled Intervention

The encoder, proposals, causal features, normalization, fitting targets,
empirical query weights and thresholds are fixed. Only the affine output
is solved. This controls final-output optimization, not encoder capacity
or the changing-encoder AdamW path. Decoupled weight decay is not part
of this least-squares data objective. Lower fitting loss is a numerical
result, not evidence of selected-set calibration or deployment safety.

## Selected Risk Error

Values below are pooled available-label sums in local image units across
repeated contexts. All/easy are two distinct columns. Unknown-label bounds
remain in fitting and original evaluation, but are never called observed
outcomes. These sums are not independent test samples or risk ratios.

| Scope | Head | Set | Known rows | Unknown rows | Predicted all/easy excess | Actual all/easy excess |
|---|---|---|---:|---:|---|---|
| fit | mean | proposals | 48061 | 1274 | 393260.344388, 193777.584695 | 235500.783506, 45001.300410 |
| fit | mean | admitted | 11813 | 71 | -48393.542387, -47537.178190 | 5212.361480, 2000.147472 |
| fit | readout | proposals | 48061 | 1274 | 359549.803319, 171096.718289 | 235500.783506, 45001.300410 |
| fit | readout | admitted | 13616 | 91 | -69350.244655, -66343.177724 | 2701.506722, 2486.750053 |
| held_SOURCE | mean | proposals | 78665 | 1444 | 421363.823969, 309281.105942 | 63859.061813, 37718.028649 |
| held_SOURCE | mean | admitted | 34888 | 504 | -238686.556342, -263057.237617 | 17382.627249, 13115.115846 |
| held_SOURCE | readout | proposals | 78665 | 1444 | 411408.113596, 309121.122754 | 63859.061813, 37718.028649 |
| held_SOURCE | readout | admitted | 35207 | 513 | -252915.118574, -283825.380164 | 17154.654914, 12988.426904 |

## Which Actions Changed?

The same frozen proposal is either admitted or rejected; trajectory and
amplitude cannot change. Benefit is positive error reduction, harm is
positive error increase. Neither cancels the other in the risk gate.

| Scope | Flow | Known rows | Unknown rows | Known benefit | Known harm | Net benefit | All excess | Easy excess |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fit | retained | 10840 | 66 | 151155.109649 | 16272.521587 | 134882.588062 | 4151.687277 | 1772.273085 |
| fit | newly_rejected | 973 | 5 | 7097.313518 | 1798.951158 | 5298.362360 | 1060.674202 | 227.874386 |
| fit | newly_admitted | 2776 | 25 | 116329.893198 | 6052.765455 | 110277.127743 | -1450.180555 | 714.476968 |
| fit | still_rejected | 33472 | 1178 | 417404.618782 | 269574.627081 | 147829.991702 | 231738.602581 | 42286.675971 |
| held_SOURCE | retained | 34292 | 490 | 256814.991906 | 36695.042204 | 220119.949701 | 16824.632588 | 12772.042564 |
| held_SOURCE | newly_rejected | 596 | 14 | 4559.416687 | 909.690790 | 3649.725897 | 557.994661 | 343.073283 |
| held_SOURCE | newly_admitted | 915 | 23 | 6341.322366 | 927.383272 | 5413.939094 | 330.022326 | 216.384341 |
| held_SOURCE | still_rejected | 42862 | 917 | 321595.575846 | 68605.773671 | 252989.802175 | 46146.412238 | 24386.528462 |

## All Registered Contexts

| Context | Completion lower vs mean pp | Original all/easy violations | Refit all/easy violations | Expand |
|---|---:|---|---|---|
| eqmotion_fold1_seed17_critic0 | -0.029944 | 9/9 | 9/9 | False |
| eqmotion_fold0_seed17_critic1 | -0.623758 | 2/4 | 2/4 | False |
| eqmotion_fold0_seed17_critic2 | 0.548834 | 2/2 | 2/2 | True |

## What The Result Establishes

Held known floor gain moves from 4.091202% to 4.323320%, a 0.232118pp descriptive change.
The direct completion-lower change is 0.048443pp, but two of three registered context lower contrasts remain negative.
Held all/easy violations stay at 13/15; full risk/support stays 0/17.
Fitting all/easy violations rise from 15/15 to 17/18 across 24 repeated views.
Thus final-affine optimization can recover a small amount of mean utility
in this pilot, but it does not fix admission risk. This does not establish
statistical significance, disprove end-to-end representation learning,
or prove that additional causal information cannot help.

## Interpretation Limits And Next Decision

A least-squares solution makes the weighted residual orthogonal to its
linear design columns. The nonlinear admission indicator is not generally
one of those columns. It therefore does not establish zero residual or
safe positive harm on the model-selected subset, even on fitting data.
That algebraic distinction is not a proof of an irreducible data limit.

Compare fitting and held-source error before attributing failure only
to domain shift. If output optimization leaves fitting admission error,
the next repair needs decision-relevant conditional information or
conditional-risk supervision, not merely more steps on this last layer.
Any such repair needs its own prospective model specification and must
be distinguished from the already-failed admission penalties and event
heads. No retrospective threshold search is justified by this table.

Unobserved future labels and absent reference mass remain separate limits.
The original full-sample support obstruction is not fixed by a smaller
regression error. No identity certification, revised threshold or deleted
empty control is allowed. All old negative studies remain negative.

No new H, independent significance, physical safety, metric, seconds,
human-gold, true 3D or foundation claim. Stage5C and SMC remain off.
