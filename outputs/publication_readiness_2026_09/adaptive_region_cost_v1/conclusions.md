# Adaptive Fitting Emphasis Does Not Repair Scene-Wise Protection

## Material Passport

Completed development experiment, 2026-09-22. Registration commit `6e4bfb88`
preceded all real fitting. Twelve small Torch cost heads are fresh_run. Frozen
forecasts, preprocessing and previous objective/region controls are
cached_verified. Independent calibration, confirmation and deployment remain
not_run. The four SDD sites were already research-design exposed.

Primary task: eight observed/twelve predicted annotation steps, stride12,
annotation pixels. Raw-frame t50 is a separate supplement, not the endpoint here.
No metric, seconds-level, sensor-time, true3D, foundation or safety-certification
claim. Stage5C and SMC remain off.

## Intervention and Compute

The single change was updating the same fourfold fitting emphasis after every
500 updates using current-model causal selections. Both arms begin with the same
frozen weights. Forecasts, 356features, 45954parameter heads, seeds17/29/43, draws,
12000updates/head, loss, optimizer and inference rules remain fixed. Selection
membership uses predictions, forecast disagreement and past motion, not future
labels. The known training-support mask controls supervised fitting only.

All twelve fits completed144000updates and36864000draws; no unknown-supervision
row was drawn. There were276refreshes,23/head. Recorded head-fit time is177.98s,
including36.91s of refresh inference; this excludes upstream predictor fitting,
loading, dependency hashing and readout/verification. Same update budget is not
same FLOPs. This is downstream-head training, not another full M3W training run.

## Primary Result

| Metric | Frozen-region head | Adaptive-region head |
|---|---:|---:|
| Equal-site ADE improvement over causal CV | 4.09764% | 4.03014% |
| FDE improvement | 4.41091% | 4.31490% |
| Hard improvement | 4.25956% | 4.09887% |
| Positive-easy degradation | -0.57490% | -0.96935% |
| Complete exact-zero-CV harmed query/seed instances | 0 | 0 |
| Selected query/seed instances | 29668 | 27740 |
| Selected instances without any ADE label | 423 | 375 |
| Selected instances with incomplete futures | 4194 | 3709 |

The primary paired ADE-gain difference is **-0.06749 percentage points**,
3000-scene-bootstrap CI **[-0.19875,+0.09748]**. It does not establish superiority
or inferiority, and fails the prespecified positive-lower-bound requirement.
The adaptive gain over CV has CI[2.42595%,5.91586%]. These resamples use only four
explored physical sites; they are developmental uncertainty, not independent
population confirmation. Seeds average errors, not predictions into an ensemble.

Negative easy degradation means improvement. Overall easy improves, but local
protection still fails:

| Site | Seed17 easy degradation | Seed29 | Seed43 | Seed-average |
|---|---:|---:|---:|---:|
| coupa | -7.81043% | -7.02188% | -7.80139% | -7.54457% |
| deathCircle | 3.41605% | 2.49304% | 2.97869% | 2.96260% |
| gates | 2.20393% | 0.20071% | 1.32630% | 1.24365% |
| hyang | -0.42863% | -0.63276% | -0.55579% | -0.53906% |

Every deathCircle seed and gates seed17 exceed2%. The joint development gate
fails on both the primary contrast and scene/seed protection. No new deployment.
The previous frozen-region head also failed protection; retaining it as a
comparison does not make it a certified or newly deployable model.

## Accuracy Versus Intervention Volume

| Policy | ADE gain | Hard gain | Easy degradation | Exact-zero harms |
|---|---:|---:|---:|---:|
| Net benefit minus harm | 11.99438% | 14.31823% | 22.04487% | 21 |
| Fixed strict policy | 4.03014% | 4.09887% | -0.96935% | 0 |
| Fixed matched intervention count | 5.52779% | 8.66917% | 6.34172% | 0 |

Strict switching is5.26108% of175756queries times three seeds. These overlapping
queries and repeated seeds are not527268independent observations. Unknown outcomes
are part of the incomplete count and must not be added to it. Partial full-grid
gain lower bounds remain negative for gates in all three seeds.

At equal intervention count, adaptive-minus-frozen ADE gain is+0.02152points,
CI[-0.12910,+0.13003], with easy degradation6.34%. There is no demonstrated ranking
advantage. The positive strict comparison against the earlier intermediate head
(+0.30122points) is secondary and cannot replace the failed primary comparator.

## Conditional Harm Diagnosis

All following cost means use complete outcomes. Old and new costs/MSE in a row
use the same selected population; comparisons across separately selected regions
are descriptive and must not be interpreted as randomized population effects.

| Population/region | Adaptive harm underestimated, of12 | Adaptive MSE lower than frozen, of12 |
|---|---:|---:|
| Fitting/all | 0 | 8 |
| Fitting/frozen-head selected region | 4 | 6 |
| Fitting/adaptive-head selected region | 6 | 7 |
| Held-source/all | 6 | 8 |
| Held-source/frozen-head selected region | 12 | 8 |
| Held-source/adaptive-head selected region | 12 | 6 |

Updating emphasis changes learning but leaves held-source conditional optimism
in all12views. For newly selected deathCircle complete rows, adaptive predicted
mean harm is0.956/0.914/0.768pixels versus observed2.263/2.852/2.571pixels. These
are complete selected populations, not easy-only values. The broad overall fit
can be conservative while the chosen population remains optimistic.

All heads refreshed23times and retained nonempty late fitting selections (last
refresh5465..9240rows across views), so this is not an unexecuted refresh or a
complete-abstention result. Selection size changes do not by themselves establish
unstable optimization; snapshots are retained for further diagnostics.

This rejects the claim that this particular lag repair is sufficient. It does
not prove lag never matters, that all possible reweighting fails, or that larger
models cannot help. The experiment gives no evidence to relax safety tolerance,
tune held-scene thresholds or hide the failed easy slices.

## Priority Gaps and Next Action

1. Reliable intervention still needs out-of-fitting conditional harm evidence.
   Same-sample emphasis alone did not supply it. Do not spend the next iteration
   sweeping refresh periods on these held-source outcomes.
2. Independent calibration and confirmation are still absent. The previous
   provenance audit rejects recycling existing heads as independent calibration;
   source-role decisions or genuinely new admitted scenes remain necessary.
3. Scene-joint decision benefits remain unproved under the completed matched
   comparison. Keep that null and the unsafe neural/cost controls in the paper.
4. The method contribution and independent generalization evidence remain below
   submission-candidate requirements. Paper preparation is not goal completion.

Source-bound analysis SHA256:
`d1c06439e16a12fa339a72158820a8df9be4a1dfdb06da3a9d2523f61aca620b`.
See [execution and verification](execution_notes.md) for reproducible commands.
