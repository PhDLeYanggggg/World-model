# Motion-Frame Repair Improves Regression, Not Safe Admission

## What Changed

Following the negative positive-tail probe, this study changes only the causal
motion representation. The trajectory bank, signed all/easy budget-excess
labels, output units, source/query weights, target RMS, fitting exclusions and
decision threshold are unchanged. history121 uses ego-history-normalized
features plus log(ego scale/context scale); history120 omits that scalar.
Sixteen new forest fits complete against eight verified original120 controls.

This is a representation-control experiment, not new trajectory dynamics,
a novel learning algorithm or independent confirmation. The primary has121
features versus120 in the control, with the same tree-count/depth/leaf limits.
One extra scalar also changes the random feature-split sequence. One producer
seed and one forest seed do not isolate a universal normalization effect.

## Primary Result

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions | All admissions |
|---|---:|---:|---:|---:|
| 0 | +9.825349% | +0.115156% | 0 to0 | 1 to7 |
| 1 | +0.709317% | +0.053858% | 1 to6 | 10 to13 |
| 2 | +30.215682% | 0.000000% | 0 to0 | 0 to0 |

Decision-cost reductions are far below the fixed1% information-screen target;
context1 also increases source-macro harmful-admission cost from4.29845e-7 to
1.40288e-6. Screen0/3: no neural expansion or deployment promotion. In context2,
all38,579known proposed decisions remain rejected despite a30.2% regression
improvement. That is a clear separation between average risk prediction and
decision utility, not evidence of a trajectory gain.

The secondary history120 arm changes decision cost by-0.012866%,-0.062795%,0%.
It also improves risk regression in all contexts, but is not a posthoc winning
primary. Its context1 harmful admissions increase1to8. All negative and
source-specific results remain in the full table.

## What The Mechanism Check Supports

The outer-neighbor geometry really controls the old scale on95.745-100% of
proposed causal rows within these sources. Median ego/context ratios span
0.227822to0.751991. The hypothesis was therefore concrete and causally testable,
not a generic instruction to normalize again. Tests verify that a change to
the old context denominator cancels from the recomputed ego coordinates while
time and masks remain unchanged; the ratio explicitly preserves that context.

The experiment supports a limited conclusion: this representation change can
reduce source-held riskMSE under the fixed learner. It does not establish that
neighbor normalization was the unique failure cause, that source shift is
fixed, or that the changed representation yields safe trajectory interventions.
Camera motion, producer errors, annotation support and unresolved future
ambiguity remain separate possibilities.

## What To Do Next

Do not respond with another scale, forest-depth, amplitude or threshold sweep.
The two completed studies show that concentrated tail cost and lower riskMSE
are insufficient proxies for the decision of interest. The next narrow task
is to trace where regression improvement occurs relative to the admissible
boundary and to the original joint deployment functional, using existing
admitted fitting outputs only. In particular, compare selected easy-reference
mass and positive harm, not just rowwise zero-crossing counts. This information
screen is a pointwise diagnostic, not the full scene-level allocator; the two
must not be conflated. A failed diagnostic cannot prove every joint allocator
would fail, and a passed diagnostic would not certify safety either.

Choose a new model-level repair only after distinguishing boundary error,
selected-reference support and candidate-forecast error on those fixed records.
Do not automatically expand this failed feature branch into a neural retrain.
No new held split, reserved outcome or relaxed support rule is authorized by
this analysis. The existing independent-confirmation gap remains open.

## Engineering Failure Kept Visible

Initial pilot PID43180 trained and saved a checkpoint, then failed exporting a
NumPy target-scale vector to JSON. The serialization-only repair is documented
in serialization_repair.md and registration_v2.json. The original checkpoint
and registration remain; pilot --resume reuses the model and verifies its full
repeat. No scientific parameter or threshold changed. All24full refits and
identical target scales are verified separately from the negative study result.

## Claims Not Made

No policy deployment upgrade, independent scientific confirmation, recovered
missing labels, metric/seconds/true3D/foundation/JEPA claim or submission pass.
Original2% all/easy positive harm, signed easy-net, reference/support and
empty-control NA remain. No identity exemption. Original24group primary is
still not estimable, not passed. Native obs8/pred12 image coordinates, one
producer seed, dependent windows and partial labels remain limitations.
Stage5C and SMC stay disabled.
