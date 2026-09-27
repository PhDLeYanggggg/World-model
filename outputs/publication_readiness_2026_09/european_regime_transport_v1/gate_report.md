# Gate Report

## Execution Evidence

| Check | Result |
|---|---|
| All 1,728 fitting cells supported | Pass; not a power claim |
| Native 2,000-update retraining control | Exact parameter match |
| 864 crossed heads / 1,728,000 updates | Complete |
| 864 matched within-regime sampling checks | Pass |
| Predictions frozen before held scoring | Commit 23b75ca6 |
| 144 source-held views / 13,824 direct MSE checks | Complete |
| Unknown-label gradient draws | Zero |
| Independent-role access | None |

The final replay status is recorded in verification.json when available.
Execution checks are not counted as scientific successes.

## Registered Scientific Screens

| Screen | Required evidence | Result |
|---|---|---|
| Cut explanation | Six favorable full-input primary CIs at both regimes; no negative/missing guards | Fail: 1 positive, 1 negative, 4 overlap at either regime |
| Larger fitting regime | Six favorable CIs at both fixed cuts; no guard harm | Fail: 4/0/2 at cut2, 1/0/5 at cut3; a cut3 coverage guard is negative |
| Size-matched magnitude | Native two-site scaled/raw 6/6 favorable primary CIs and no guard harm | Fail: 3/0/3 primary; 6 negative coverage CIs, 1 negative capture CI |
| Raw interaction consistency | Six same-signed nonzero full-input CIs | Fail: 1 positive, 5 overlap |
| Scaled interaction consistency | Six same-signed nonzero full-input CIs | Fail: all 6 overlap |

Counts use the unchanged outer three-site target, with dependent replicas
and seeds averaged before the four-locality bootstrap. They are descriptive,
unadjusted source-development intervals. The different-target diagnostic
cannot substitute for the primary endpoint. Cost-only mechanisms do not
establish a true-auxiliary failure cause.

## Unchanged Boundaries

No model promotion, new policy evaluation, independent confirmation or
submission-readiness claim. Stage5C execution and SMC activation remain false.
No metric, seconds-level, human-gold, true3D, foundation or physical-safety
claim. Protocol: obs8/pred12 native steps in detector pixels.
