# Severity Auxiliary: Completed, Cost Gate Failed

## Material Passport

Fresh_run: fitting-support audit, 144 native-Torch heads /288,000 updates,
36 result groups /144 source-held folds, three-seed locality contrasts and
3,000 bootstrap resamples per assignment. Cached_verified: source producers,
causal features and288 matched prior heads. Not_run: independent selection,
reserved calibration, confirmation, new trajectory policy and deployment.

Registration c3d74244 and support176640d7 preceded fitting. All predictions
were hashed and frozen in41a478ca, pushed before the new outcome readout.
Training PID42824 exited normally. Summed fitting273.4563seconds excludes
loading, inference and verification. Unknown-label draws0. CPU4/interop1/
workers0, checkpoint/resume; no remote jobs submitted or modified.

## What Changed

Replace ordinary easy-membership BCE with H/mean_fitting_H-weighted BCE.
Everything else is matched: architecture, initialization, sampler, cost
losses, optimizer, budget and cost decoding. Future H/E remain detached
supervision; the auxiliary output is never multiplied into predicted costs.
It estimates a harm-weighted fraction, not ordinary easy probability.

## Registered Result

Full-input, positive-disagreement easy-harm MSE contrasts:

| Comparator | Positive CI | Negative CI | Overlap | Point range (%) |
|---|---:|---:|---:|---:|
| Original cost model | 0 | 1 | 5 | -20.1401 to +2.3801 |
| Matched cost-only | 0 | 1 | 5 | -20.1401 to +2.3801 |
| Ordinary membership auxiliary | 0 | 0 | 6 | -11.5653 to +1.0416 |

Original and matched-control predictions agree exactly; these are required
controls, not independent replications. No missing-support contrast was
dropped. Tail guards also fail: compared with original/control, top10 harm
capture has one negative interval and coverage error has one negative
interval. Compared with ordinary auxiliary, top10 capture has three negative
intervals and coverage has one. All-harm MSE has no negative interval in the
full comparisons, but that neither rescues easy-harm nor proves equivalence.

Motion-only remains negative evidence:0 positive,1 negative,5 overlapping
easy-harm intervals vs original/control;0 positive,2 negative,4 overlapping
vs ordinary auxiliary. Its worst source point is -48.69% vs original.

## What Was Learned, and What Was Not

Full fixed-batch median normalized cost loss falls0.65457 to0.47562; weighted
BCE falls0.36319 to0.08788. Optimization happened, but lower fitting loss does
not establish transported risk estimation. Compared with original,42/72 full
fitting views improve easy-harm MSE;25 of those do not improve held MSE.
Only17 improve both, and34/72 improve held MSE overall. Against ordinary
auxiliary,50/72 improve fitting MSE,30 are fitting-only, and27/72 improve held.
These dependent counts are diagnostic, not independent statistical wins.

Harm-event ranking improves in some comparisons: full positive-disagreement
AUROC has4 positive intervals vs original. Yet easy-harm magnitude has none,
and tail capture can worsen. Recognizing the event is not estimating its
cost or certifying a safe policy. The weighted supervision repair is not
supported by the registered endpoint.

## Decision and Evidence Limits

Do not promote this head, change thresholds or open independent roles.
Keep the original frozen cost/forecast chain unchanged; this is not a claim
that the existing chain is independently deployment-validated. Historical
Stage35/37/43/44 exposed scores remain exploratory.

Three seeds are averaged within each locality before resampling four
localities per assignment. Reused roles and overlapping windows are not
independent. CIs are exploratory and not multiplicity-adjusted. Engineering
replay and test evidence is recorded separately in verification.json; it
does not reverse the failed scientific gate.

Next: frozen recording/track-level error attribution and fitting-only weight/
gradient concentration, retaining all source roles. Distinguish concentrated
severity support from broader feature-conditional transport failure before
another fit; do not start coefficient, clipping or threshold sweeps.

8 observed/12 predicted annotation steps; detector-derived pixels only.
No metric/seconds, human gold, physical-safety, true3D, foundation, new
trajectory gain or submission-ready claim. Stage5C/SMC off; goal ongoing.

[Full results](results.md), [failure analysis](failure_analysis.md),
[loss curves](training_loss.svg), [all-source contrasts](paired_contrasts.svg).
