# Easy Occurrence and Conditional-Risk Repair

Status at registration: not_run. This is a development experiment, not a new
independent test. The original incomplete primary is not replaced.

## Question

The previous uniform risk offset removed useful switches and did not satisfy
the selected-harm screen. Its easy-risk axis participated in most rejected
admissions. The old four outputs were score-basis components: only two linear
combinations were supervised. They must not be interpreted as easy occurrence
probabilities or identified conditional cost moments.

Test whether explicit occurrence and conditional-cost supervision improves
easy-risk prediction and joint decisions, holding the forecaster, floor,
all-risk score, causal inputs, utility, eligibility and 2% budget fixed.

## Fixed Paired Training

108 existing groups; 3 forecast seeds; 12 already-opened source-training
development localities. Each role design remains 4 producer / 4 controller /
2 fitting / 2 held development sources. Independent selection, calibration
and confirmation remain closed. No new future-target information is input.

Both arms have the same 64-wide SiLU MLP and three raw outputs, initialized
identically. Inputs are the existing causal features and six causal rollout /
motion descriptors, standardized with the inherited fitting-only normalizer.
pi=sigmoid(logit), R=softplus(reference), H=envelope*sigmoid(harm),
easy_signed=pi*(H-.02*R). The envelope is causal rollout disagreement.
All-risk remains the verified frozen subset_aggregate score.

- marginal: .5 query-balanced row squared signed error + .5 squared query-mean
  signed error, against I_easy*(observed_harm-.02*floor_error).
- supervised: the same marginal loss + query-balanced binary cross entropy
  for I_easy + easy-conditional .5*(reference MSE + positive-harm MSE).

Each auxiliary coefficient is fixed at one. Conditional costs use the existing
fitting cost scale. This is one prespecified supervision-package contrast, not
an isolated causal test of each loss. Same queries, updates and initialization;
32 source-balanced queries/update, 2,000 updates/head, AdamW lr=.0003,
weight_decay=.0001, gradient clip5, CPU4/interOp1/workers0. No checkpoint selection
on held labels. Use the final registered update. Record all monitor losses.

Easy uses the existing label rule 0<CV_ADE<=the frozen easy cut. Zero-CV rows
are not easy; unknown outcomes never enter fitting. Each fitting pair must have
easy labels; absence is an explicit unsupported-condition blocker, not invented
supervision. First audit support, then run a100-update paired resource pilot;
resume the same fits if storage projection retains10GiB. Keep compact atomic
checkpoints with optimizer/RNG/sample-chain and immutable input references.

## Frozen Action Controls

For raw aggregate risk and each new arm, evaluate independent admissions and
joint utility allocation at that arm's per-query admission count. Also use
the intersection of all three independent admission masks as a common feasible
anchor; all three matched joint policies have exactly the same count per query.
Primary exploratory contrast: supervised_matched versus marginal_matched.
Secondary: both new arms versus raw_matched and their own-count policies versus
raw_joint. Empty common anchors remain empty; no count adjustment after readout.
Use the existing joint solver, node_limit256; numerical failure returns its
feasible anchor. Sigmoid probability cannot change the sign of individual
conditional risk; it can change aggregated risk weights. Report this limitation.

All108 training groups and causal action arrays must be committed before the
new held-development cost readout. No threshold tuning, held-label early stopping,
selecting only favorable localities, dropping undefined risk or winner deployment.

## Evaluation and Screen

All arms retained. Report query-balanced easy Brier/log loss/conditional errors,
ADE/FDE, hard/easy, selected positive harm, full-floor-denominator harm, tails,
unknown labels, zero-CV harm and intervention counts. Use paired locality
reductions and 3,000 nominal source-locality bootstrap draws seed101531,
including all12localities. Repeated source-role/seed contexts are dependent.
These are exploratory intervals, not independent confirmation or simultaneous CIs.

Exploratory screen requires positive matched ADE CI, positive full-denominator
harm-reduction CI, every selected-risk view defined and<=2%, easy gain versus
CV>=-2% in each view, and no zero-CV harm. It is deliberately not a calibration
certificate. Undefined selected denominators fail the screen, not silently zero.
No new deployment follows from a development screen alone.

## Interpretation

The expectation factorization is elementary, not a novelty claim. BCE/Brier
are proper-scoring motivations, not evidence of finite-sample calibration or
shift robustness. Primary reference: Gneiting and Raftery (2007), JASA102:359-378,
doi:10.1198/016214506000001437, finite/binary scoring-rule examples in Section3.

Image-local detector-silver trajectories, obs8/pred12, raw-frame stride12 only.
No metric, seconds, human-gold, physical-safety, true3D, foundation or submission
readiness claim. Future labels are loss/eval only, no central velocity, no test
endpoint goals. Stage5C execution and SMC remain disabled. Historical exposed
Stage35/37/43/44 results remain exploratory and are not recertified here.
