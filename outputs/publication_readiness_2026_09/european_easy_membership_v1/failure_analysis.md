# Membership Success Does Not Resolve Expected Harm

## Evidence Scope

Fresh_run diagnostic classifiers and evaluation; cached_verified source
lineage and parent forecasts. No new policy, trajectory, risk calibration or
independent confirmation. The positive finding is restricted to full-input
easy-membership prediction on source-development localities.

## Failure Taxonomy

| Candidate explanation | Observation | Interpretation / limit |
|---|---|---|
| No usable easy-membership signal | Full MLP passes all three registered metrics in all six assignments | Not supported as a blanket explanation for the earlier expected-harm failure |
| Apparent gain is only prevalence correction | Full MLP retains six positive Brier and log-loss intervals against training conditional prevalence | This confound does not explain the full result; it matters strongly for motion-only |
| Linear probability tails are fragile | Full linear log-loss interval includes zero for producer0/controller2 despite positive Brier | Keep the failed guard; do not select another metric after readout |
| Full and motion-only conditional tasks are equivalent | Conditional easy prevalence medians are 0.17565 and 0.04946 respectively | Their disagreement sets differ; do not call the contrast a causal modality ablation |
| All held localities are well calibrated | Full MLP ECE ranges from 0.00726 to 0.22652 | Residual locality-specific probability error remains |
| Successful membership guarantees expected-cost accuracy | No conditional severity model was fitted in this round | Unsupported; expected-harm and policy gates remain closed |
| Slow runtime prevented learning | All 288 fits finish at 2,000 updates with zero unknown-label draws | Runtime did not prevent this diagnostic; this does not prove optimization sufficiency for other tasks |

## Adverse Motion-Only Result

The stronger training-conditional constant changes the interpretation.
Motion-only MLP Brier skill has one positive interval (producer1/controller2),
one negative interval (producer0/controller1), and four overlapping intervals.
The negative point is -7.275%, CI [-12.175%, -1.851%]. All six log-loss gain
intervals overlap zero. Linear has five negative and one overlapping Brier
interval. Reporting only the positive unconditional-constant comparisons would
hide this failure. No change to the primary full-input criterion was made.

Support is also weaker on this conditional subset: 11/72 motion-only views
have fewer than 20 positive examples, with a minimum of seven. None is removed
or relabeled. Full conditional views have no weak-support flags. Sparsity may
contribute to unstable motion-only estimates; these results do not isolate it
from the changed disagreement population or feature predictability.

On the registered constant comparison, motion-only fitting Brier improves in
all 72 views, but held Brier improves in only 66 linear and 67 MLP views.
Full improves in all 72 for both arms. Those repeated views are dependent and
are not 72 independent replications.

## What This Narrows Down

Earlier fractional and frozen-readout experiments assigned excess easy harm
to non-easy cases. A dedicated easy-membership head can separate that event
on the full inputs. It remains possible that the conditional severity is
poorly predictable, heavy-tailed, or sensitive to role/locality shifts; this
round does not distinguish those explanations. The planned factorization
must therefore be tested against matched direct-cost controls and the
original expected-cost reference, not just against a constant classifier.

No post-readout threshold sweep, probability calibration, model promotion,
Stage5C or SMC. Source-developed detector-derived pixel/annotation-step data
do not support metric, seconds, human-gold, physical-safety, true-3D or
foundation claims.
