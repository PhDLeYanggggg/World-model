# Development-only methods and results addendum

This is an anonymous development appendix, not a replacement abstract or a
submission-ready contribution. It extends the exposed European SOURCE study;
it does not combine its scores with historical SDD or Stage26/37 results.

## Method

Let the cost head output five nonnegative moments for each legal candidate.
The unchanged decoder admits positive predicted utility only if the all and
easy signed scores are nonpositive, where `q=H-0.02R`. We retain the original
five-moment/ranking objective and directional score loss. At each update, the
current maximum-utility admissible action is recomputed from predictions and
causal candidate legality and detached. On known supervised rows, channels
whose observed signed score is positive receive the additional loss

`L_admit = mean_{batch,channel} I(selected) I(q_true>0)
                 [relu(q_true-q_pred)/s_channel]^2`,

where `s_channel` is the unchanged physical fitting RMS quadrature. Unit
weight is fixed in advance. Unknown rows contribute zero observed-cost loss.
The label identifies a supervised prediction error, never the inference action.
Unlike earlier fixed teacher-region losses, the action receiving this
additional term changes with the current model. This is a directional safety
surrogate, not an expected-risk guarantee or a novelty claim by itself.

## Matched evidence

The complete design trains54heads:6contexts x3seeds x3objectives. All heads
retain30405parameters,462causal fields and2000optimizer steps. All36original
and upper controls reproduce their complete prior scientific states; the new
head has an exact2000-step refit.72source/context/seed groups,144control
prediction/action/full-metric checks, and the separate complete inference
replay are exact. CIs use3000locality resamples after context/seed averaging.

The primary admitted-minus-upper known ADE change is-0.075404pp,
95%CI[-0.245661,+0.026133]. Its wholly-unknown-row completion-lower change is
-0.120265pp, CI[-0.290321,-0.012139]. Complete recording-risk support is46/444
versus41/444for upper score, with308known easy violations and42undefined
easy views. Thus the tested repair fails both primary improvement and the
unchanged complete-policy safety requirement.

A secondary same-head action restriction supports useful neural usage:
known+0.194407pp, CI[+0.049385,+0.386839]; completion-lower+0.164782pp,
CI[+0.019557,+0.359793]. This pre-existing kind of secondary signal must not
replace the failed primary. No independent outcome was opened, no deployment
changed, and no new trajectory dynamics contribution is established.

## Claim boundary

The arithmetic diagnosis and unsuccessful controlled repair can support a
development account of selected-population risk misestimation. They cannot
support calibrated safety, an effective new main method, independent
cross-dataset performance, a true3D/foundation world model or readiness for
CVPR submission. Independent evidence, stronger method comparisons and an
effective risk-control mechanism remain missing. All coordinates and horizons
remain pixel/raw-frame, supplied detector-silver rather than human gold.
