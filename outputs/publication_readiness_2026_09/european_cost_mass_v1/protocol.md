# Fitting Cost Versus Harm-Mass Experiment

## Material Passport

Parent: crossed-regime study at commit d68ff381. Its 56 public/source bindings
and local detailed metrics were rechecked. Local arm64 runtime and previous
real Torch training are verified. CREATE was inspected read-only; three
unrelated jobs are pending, none submitted or modified. This experiment
needs cached local arrays and scalar fitting, not new GPU training.

Result roles: cached_verified neural heads, forecasts and feature lineage;
fresh_run loss decomposition, 432 two-parameter readouts and source-held
evaluation; not_run new neural training, forecasting, policy or independent
selection/calibration/confirmation. Obs8/pred12 native steps, detector pixels.
Stage5C and SMC stay off. Prior source-held exposure remains development.

## Question

The prior crossed experiment did not isolate a uniform cut or fitting-regime
repair. Even its native two-site scaling worsened all six full-input
coverage-log intervals. Existing fitting-record summaries also show smaller
MSE with much lower predicted harm mass; their row weighting differs from
the scalar fitter's locality weighting and must be separated.

We ask whether restricted origin regression and nested projection already
create a loss-versus-mass tradeoff on honest fitting-only OOF data, and whether
an equally simple projected moment-matching readout transports better.
This does not claim squared error is intrinsically wrong for a conditional
mean. Gneiting's point-forecast framework establishes its consistency for the
mean under suitable moment conditions; finite restricted regression and
subsequent clipping do not automatically preserve empirical harm mass.
[Gneiting, Making and Evaluating Point Forecasts](https://arxiv.org/html/0912.0902v2),
Sections 2.2 and 3.1; publication DOI10.1198/jasa.2011.r10138. This source
motivates a distinction, not evidence that the proposed readout will improve.

## Fixed Inputs and Methods

Reuse the honest OOF magnitude bank, not the crossed two_cut3 heads. Every
inner predictor, reference, preprocessor and label definition excludes its
own held locality; the outer locality is excluded from all fitting. The
forecast-producer roster stays disjoint from controller sites. Preserve all
144 outer views, three seeds, both full/motion families and all three arms:
cost-only, true auxiliary, shuffled auxiliary. No feature, target or split change.

For each arm/view, compare raw identity, the frozen origin-L2 readout, and
one new mass readout. Denominator columns remain unchanged. Both slope bounds
stay [0,8]. Fitting uses the same known, positive-envelope rows with equal
mass per fitting locality. There is no threshold or regularization sweep.

Fit alpha so the weighted mean of min(envelope,alpha*H_all) matches the
weighted H_all label mean. Then fit beta so the weighted mean of
min(corrected_H_all,beta*H_easy) matches the weighted H_easy label mean.
Use exactly 80 bisection steps on the fixed bounds. A zero target uses slope0.
If the target exceeds capacity at8, retain slope8 and explicitly flag the
unmatched moment. No infeasible cell is removed. These are training-only
moment constraints, not conditional, held-scene or conformal guarantees.

Store every coefficient and its input hashes before outer scoring. Do not
refit on the primary outer labels. Candidate predictions must be committed
before scoring, even though these source localities have historical exposure.

## Diagnostic Decomposition

For both harm moments and each arm/view, split weighted squared error into
zero-target and positive-target contributions, with per-locality contributions.
Record the origin-L2 numerator E[p*y], denominator E[p^2], its zero-target
share, and the unprojected mean slope E[y]/E[p]. Compare raw, unprojected L2,
projected L2 and projected mass readouts on fitting OOF records only. Record
how much nested clipping removes from predicted mass. These deterministic
identities do not establish a general causal explanation or an independent
test result. Report equal-locality and row-weighted fitting mass separately.

## Evaluation and Decision

Keep the unchanged outer three-site easy definition and the existing primary:
positive-envelope expected easy-harm MSE. Guards remain top10 harm capture,
coverage-log error and all-envelope H_all MSE. No ADE/FDE, easy-degradation
or new trajectory-gain claim follows from these cost-estimation quantities.

Report mass versus raw and L2 in every arm; true versus cost-only and shuffled
under the matched mass readout. No favorable arm or assignment selection.
Average three seeds inside each locality and use 3000 paired resamples of
four localities per assignment. Six overlapping assignments yield descriptive,
unadjusted intervals, not six independent datasets. Missing support is retained.

Primary readout screen: full-input cost-only mass beats both raw and L2 with
six favorable primary CIs each, no negative/missing guards, and supported
fitting moment constraints. Auxiliary-information screen separately requires
true mass to beat both cost-only and shuffled mass with six favorable primary
CIs each and no negative/missing guards. These screens do not authorize
deployment or independent-role access. A failed primary cannot be replaced by
coverage improvement alone. All failures and bound hits remain visible.

## Boundaries

No future endpoint or label at inference, central velocity, test goals or
test normalization. No independent selection, reserved calibration or
confirmation access. No metric/seconds, human-gold, true3D, foundation or
physical-safety claim. No new latent generation or SMC. If the tradeoff
persists, report it rather than relaxing the primary or protection criteria.
