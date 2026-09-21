# When to Trust Neural Motion Forecasts: Baseline-Relative Joint Intervention for Multi-Agent Forecasting

Working draft, evidence reconciled 2026-09-21. Method proposal with completed three-seed development
experiments, including a matched Transformer/EqMotion study and a completed
baseline-relative output ablation. The latter reduces drift but yields only
tiny guarded development gains. None supports a new deployment or the proposed
joint-intervention contribution claim.
The completed v6 input-conditioning protocol retains the repaired v5 source population.
The older results below retain their original, conditional observation population.
The latest fixed-candidate diagnostic limits recoverable gain from selection and
whole-path scaling. It does not supply a positive learned method result. This is
an evidence-bearing working manuscript, not a submission-ready paper. A subsequent
readout/loss factorization improves fitting but worsens held-scene forecasting.
The motion-to-start probe finds only one-direction probability transfer. The
subsequent SDD geometry/image bridge is an input prerequisite. A separately
approved matched auxiliary experiment now completes 54 fresh Torch fits: source
pretraining reduces neural degradation, but no fit beats CV or preserves easy
cases. A further 54-fit control study separates source supervision from main
training exposure: correct source pairing has no stable demonstrated advantage
over source-label permutation. This revision retains both negative studies; it
does not turn the older diagnostic bridge into a training result.
A subsequent 27-fit observed-unit conditioning experiment measures actual
gradient response and repairs a unit-dependent rollout feature, but no new fit
passes easy preservation. Balanced internal-loss gradients correspond to worse
primary forecasts, not a successful contribution.
A further45-fit source-supported start-classification study fails bidirectional
transfer. Its Hotel improvement comes from a mean-probability shift; source and
mixed models are worse than a constant source prior there and worsen ETH. This
probability diagnostic does not establish better trajectories or a new policy.
The subsequent source-internal five-site visual comparison also completes all
30 fits and is negative. It narrows the diagnosis: failure is not confined to
source-to-main transfer, and annotation-change frequency is not forecasting cost.
The latest frozen-predictor comparison completes all 24 coupling controls at the
same reference intervention count. Joint versus unary-geometry ADE is identical
in 21 combinations, slightly better in two and slightly worse in one. None passes
easy preservation. This does not establish the proposed coupling contribution.

Evaluation amendment, 2026-09-21: the author delegated the pending metric choice.
Native-coordinate ADE/FDE with equal mean of within-scene ADE gains is adopted
for subsequent comparisons, with the old normalized scores retained. This is
post-hoc protocol development, not retrospective preregistration. A fixed first
source readout finds 28.63% future-oracle headroom but no repair of the old neural
static-history subset (-5.38% mean three-seed gain). Neither is independent
confirmation or full-population neural success. The historical abstract/results
below keep their original metric; no favorable replacement of those studies is
claimed. [Amendment and paired readout](native_metric_v1/conclusions.md).

The subsequent registered full-population loss comparison is now complete:
24 real Torch fits with identical architecture, batches and budgets. Native loss
improves source-exclusion ADE by 7.63% over CV, versus 2.18% for the old-loss
control. The direct paired contrast is 5.56% [3.34%, 7.78%]. All three seeds and
four explored sites are positive, but perfectly CV-predictable paths suffer
positive absolute harm. This establishes a useful developmental predictor
contrast, not safe deployment, independent confirmation, joint-policy novelty
or a submission-ready paper. [Complete comparison](native_forecast_v1/conclusions.md).

A follow-up completes 18 random-initialized, two-site-excluded native predictors
(three seeds,72,000updates) and physically materializes twelve clean cost-head
training views. Every row's upstream fit excludes both its own site and the
proposed outer validation site. Independent cost arithmetic and checkpoint replay
pass, but no new risk head is fitted yet. All four source sites remain
design-exposed, and this infrastructure result is not an independent performance
or calibration claim. Strict zero-reference absolute-harm protection is retained
without a new pixel tolerance. [Lineage repair and limits](native_nested_v1/conclusions.md).

## Abstract

Average forecasting gains can conceal degradation on trajectories already well
predicted by a simple motion baseline. We study baseline-relative selective
intervention: estimating benefit and harm from cross-fitted forecasts and
selecting replacements over an observed interaction graph. The task observes
eight annotation steps and predicts twelve. After diagnosing a past-scale
weighting problem, we adopt a disclosed native-coordinate ADE amendment with
equal mean of within-scene relative gains. A matched 24-fit source-exclusion
study improves ADE by 7.63% over CV with native loss, versus 2.18% with the old
loss. It does not preserve all zero-CV easy cases. The four sites are explored
development data, not independent confirmation. The historical studies below
retain their previously frozen past-normalized metric. Completed three-seed development
comparisons with causal baselines, a Transformer and fixed-head EqMotion do not
establish a joint-selection advantage. Baseline-relative output bounds reduce
drift, but ordinary regression deferral still fails the 2% easy-degradation limit.

Controlled visual-input, objective and sampling experiments investigate the
remaining failure. In the sampling comparison, 54 fresh fits and 18 verified
controls retain identical model, loss and update budgets; neither track nor event
balancing improves the complete-cohort primary endpoint. To distinguish routing
failure from limited candidate predictions, we compute future-informed diagnostic
ceilings over the frozen forecasts. Perfect choice among eight candidates per
seed gains 1.627%; perfect whole-path correction scaling increases this to 1.726%.
These are oracle diagnostics, not learned performance. Under the fixed metric,
365 static-history windows contribute 89.37% of baseline error, while current
candidates barely correct subsequent movement. This identifies a restricted
action-class limitation without asserting that new predictors cannot improve.
A further matched 2x2 readout/loss comparison completes 54 new fits: transformed
residual supervision improves fit-cohort error but damages held-scene prediction,
so output-range repair alone is not sufficient.
Past-only coordinate conditioning also fails to produce safe forecasting gains.
A separate probability probe finds a Hotel-to-ETH start signal (AUROC 0.81-0.82),
but reverse transfer remains near chance; neither establishes better trajectories.
An additional 54-fit source comparison uses 229,333 eligible windows from the
original 40 SDD training recordings. SDD pretraining improves on matched neural
controls by 0.54-0.70%, but all held-scene fits remain worse than CV and fail easy
preservation. A further 54-fit mechanism study adds a main4k control and source
label permutation. Reduced main exposure itself improves the neural controls;
correct versus permuted source gains are +0.118%, -0.015% and +0.058% across
input variants, with all descriptive site intervals crossing zero. No new
fit meets the easy-preservation limit. Past RGB does not consistently improve
on image-coverage masks.
Previously explored data remain exploratory; independent confirmation and a
positive contribution are still missing. We make no physical-time, metric-safety
or general world-model success claim.

## 1. Introduction

Strong causal predictors can be difficult to improve consistently. A neural predictor may capture turns or interactions while performing worse on straight, low-variance motion. Replacing the baseline everywhere therefore asks a different question from deciding where the neural model has useful information. The latter question is especially relevant under scene shift, where a global average provides limited guidance about the risk of a local intervention.

Multi-agent prediction introduces another difficulty. Per-agent decisions can combine incompatible futures even if each predictor produces a coherent scene when used alone. We propose to treat the intervention vector itself as a structured prediction and to evaluate its excess error relative to a fixed baseline.

The study tests whether baseline-relative loss supervision and joint intervention improve mixed forecasts beyond cost-aware routing alone, and whether that effect survives dependence-aware evaluation. Regression deferral and joint compatibility already have substantial prior work. The contribution cannot be established by renaming those components; it requires matched controls, a defensible methodological distinction and new independent results.

## 2. Related Work

AgentFormer jointly models social and temporal structure with agent-aware attention (Yuan et al., ICCV 2021). EqMotion introduces equivariant motion prediction and invariant interaction reasoning (Xu et al., CVPR 2023). SingularTrajectory studies a unified representation across trajectory-prediction tasks (Bae et al., CVPR 2024). These works preclude claiming that temporal attention, relative geometry or a common motion representation are novel by themselves.

Joint Metrics Matter studies joint forecasting errors and collisions (Weng et al., 2023). The present proposal must demonstrate a benefit beyond adding a joint metric or training penalty: it concerns which predictions from competing predictors can be selected together.

Cost-sensitive expert deferral predates this proposal ([Mozannar and Sontag, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html)). [Mao, Mohri and Zhong (ICML 2024)](https://proceedings.mlr.press/v235/mao24d.html) explicitly study regression deferral with a fixed predictor. Our cost heads therefore require comparison to established deferral objectives, not only confidence thresholds. The [real two-action comparison](8to12_deferral_v7/results.md) is now complete on both frozen v7 predictors, three seeds, linear/width64 heads and two fixed cost bounds. Every result is retained without selecting a new winner. This task-specific control is not a reproduction of Mao et al.'s reported experiments.

Selective-regression work also shows that reduced coverage need not protect every subgroup ([Shah et al., ICML 2022](https://proceedings.mlr.press/v162/shah22a.html)). Our easy-error slice is a different construct, but the warning motivates reporting slice-specific damage rather than treating reduced intervention as a guarantee. The [source-scoped review](joint_intervention/deferral_and_coverage_prior_work.md) distinguishes these established results from our remaining hypotheses.

[SPO](https://arxiv.org/pdf/1710.08005v5) and
[decision-focused ranking](https://proceedings.mlr.press/v162/mandi22a/mandi22a.pdf)
already connect cost prediction to downstream decisions. Ranking or regret loss
alone is therefore not our novelty. The inspected SPO setting has a known
feasible region; our predicted-harm budget also changes the feasible actions.
Its consistency statement cannot simply be imported into this selector.

[Decision calibration](https://proceedings.neurips.cc/paper_files/paper/2021/file/bbc92a647199b832ec90d7cf57074e9e-Paper.pdf)
and [multicalibration](https://proceedings.mlr.press/v80/hebert-johnson18a/hebert-johnson18a.pdf)
make aggregate fit an insufficient novelty argument for conditional reliability.
Our extra scene context, continuous costs and outcome-defined easy labels require
careful qualification. The [focused review](conditional_decision_v1/prior_work_and_method_boundary.md)
records inspected sections and limits rather than asserting an inherited guarantee.

Conformal Risk Control and Learn then Test provide established tools for controlling losses or selecting risk-constrained policies. SODA-MPC combines conformalized OOD monitoring with reachability fallback in control. We target excess forecasting loss relative to a fixed predictor; we do not claim formal physical safety from low ADE.

[HCP](https://arxiv.org/html/2306.06342v4) requires both across-group and within-group exchangeability. [GHCP](https://arxiv.org/html/2608.15500v1) uses initial target-group labels under additional sampling assumptions; it is not a general repair for serially dependent windows. [CAFHT](https://arxiv.org/html/2402.09623v2) permits dependence within trajectories but requires exchangeability across trajectories for simultaneous path coverage. [Adaptive CP for motion planning](https://proceedings.mlr.press/v211/dixit23a/dixit23a.pdf) studies online coverage and feasible control. None of these guarantees alone orders a selected point forecast against our baseline. The [assumption audit and falsifiable claim matrix](joint_intervention/statistical_assumptions_and_claims.md) separate these tasks and specify which comparisons remain missing.

## 3. Method

### Evidence Roles and Observation Contract

The approved task is offline annotated-history forecasting with eight observed
and twelve predicted native annotation steps. Supplied historical positions may
have been interpolated using later annotation controls. Past-indexed input access
therefore does not establish strict sensor-as-of observation availability.
Explicit future targets are confined to supervised losses and evaluation; they
do not enter query features, neighbor membership or goal construction.

Development experiments have already run under approved versioned contracts.
Later repair studies use only the 11,966-window fit cohort with ETH, Hotel and
grouped Zara physical-site folds. Students/development, independent calibration
and confirmation labels remain closed for those repairs. A held fit fold is not
a previously untouched test site after repeated exploratory comparisons. The
independent-confirmation protocol remains unresolved; this is distinct from
the approved and completed development work. The SDD bridge described below
retains diagnostic-only status. A separate user-approved source contract admits
the original 40 SDD training recordings at stride 12 into the matched auxiliary
experiment below. No admission is inferred merely from a successful input check.

A [full SDD support census](sdd_state_support/conclusions.md) further separates
row count from state-change support. At diagnostic stride 12, 249,384 complete
pedestrian windows contain 328 fixed static-to-movement proxy windows from 78
recording-local track IDs and 84 disjoint spans. Stops and turns have broader
support. The intervals define different raw-frame tasks and are not compared as
prediction scores. Neither track IDs nor disjoint spans certify independence.
We also distinguish directly sampled source controls from unsampled controls
inside an observation interval, avoiding an aliasing-based claim of missing
annotation support. This source diagnostic admits no auxiliary training role
and does not establish that additional data improve forecasting.

An additional [SDD past-image input check](sdd_past_images/conclusions.md) follows
the source correspondence repair. It retains geometric image support separately
from an inferred current-frame black-border mask, without erasing partial or
short histories. All 60 fixed 64-frame prefixes are processed; 180 independently
decoded frames produce 2,074 exact crop replays. The 302 non-lost crops affected
by suspect borders illustrate why image extent alone is inadequate. Original
observed pixels remain available because dark border regions can be real image
content. These masks are not human labels or person-visibility segmentation.
This diagnostic admits no new training/evaluation role and proves no forecasting
benefit; offline interpolation and unknown physical timing remain limitations.

Use the observed agent histories, legal past-only context and candidate rollouts to estimate gain and harm. Train these heads using cross-fitted predictions within training recordings. Freeze both the floor and candidate predictor before training/calibrating the deployment gate. The gate minimizes estimated excess loss over agents plus an interaction penalty on the mixed forecast, subject to supported risk budgets.

A past-context Transformer encodes observed history and neighbor tokens; queries
contain requested prediction times and baseline rollouts, never ground-truth
future coordinates or validity masks. The completed v1 experiment uses normalized
coordinate MSE and a fixed update budget. For error functional L, baseline B,
candidate N and target Y, define g = L(B,Y) - L(N,Y), benefit = max(g,0), and
harm = max(-g,0). Paths are normalized using observed context only. Separate
nonnegative costs imply predicted gain = benefit - harm, an internally coherent
prediction, not calibrated safety. Scene-held-fold predictions supply cost
targets; every producer and fitted ancestor must exclude the whole held fold.
Ridge and neural heads use the same 11,966 OOF examples and fit-only preprocessing.
Neither has established a predictive advantage in the first real experiment.

The [backend checks](supervised_backend/implementation_and_limits.md) verify
synthetic CPU/MPS recovery and real-input invariance under future corruption.
The [new real experiment](8to12_development_v1/results.md) supplies separate
development evidence. The public-core and matched realized-count comparisons
are now complete in v6, without establishing a contribution. Independent risk
calibration remains pending.

The [neural cost-head path](neural_cost_head/implementation_and_limits.md) now also fits nonnegative benefit/harm regression on the same verified OOF inputs as the ridge control, with fit-only normalization and unchanged candidate forecasts. This enables a capacity-controlled comparison rather than attributing gains to an untrained interface. Synthetic CPU/MPS recovery is verified; its development fixture selects the floor, and apparent intervention occurs only on queries lacking complete labels. Neither a real neural advantage nor calibrated safety follows from this implementation.

The [development evaluator](development_evaluation/implementation_and_limits.md)
compares floor, uncontrolled, budget-constrained unary, scene-uniform and
pairwise-joint controls on identical candidate forecasts. Scene membership uses
past observations, not future-label completeness. Raw errors remain recording-local;
the current normalized summary equally weights physical scenes. Common budget
caps do not establish equal realized coverage. The first real comparison is now
complete and negative; the subsequent v6 matched-count mechanism test also fails
to show stable improvement.

A [frozen-policy calibration interface](risk_calibration/implementation_and_limits.md) now consumes those exports without refitting. It requires a prespecified family, order, risk functional and scene aggregation. Missing-label interventions receive the worst bounded loss instead of being excluded. For bounded [0,1] scene losses, its current reference screen adds sqrt(log(M*K/delta)/(2*n)) to the empirical mean for M policies and K risks, using n physical-scene clusters. An unchanged baseline has analytically zero excess risk; a learned policy that happens not to intervene on the observed sample still requires a sampling bound. These are conditional statistical statements, not evidence that the actual scenes are IID, a novel risk theorem, a 2% relative easy-error guarantee, or physical safety. Only synthetic integration has been executed; no real calibration result is available.

A binary optimizer solves this decision with MILP, using past-only coordinate
restoration, baseline-relative pair costs and explicit predicted-harm/intervention
budgets. Its real v1 development comparison shows no gain over independent
selection. The [implementation specification](joint_intervention/method_and_checks.md)
separates this mechanism from the statistical assumptions. JFP already studies
unary/pairwise forecast compatibility and heuristic overlap penalties; joint
optimization alone is not a novelty claim. Its proposed value still depends on
matched comparisons and independent evaluation. See the
[focused related-work audit](joint_intervention/related_work_constraints.md).

An additional [exact-count control](matched_coverage/method_and_limits.md) sets the joint intervention count to the independent policy's count on each observed query, before labels are read. It retains the same forecasts, support and predicted-harm cap. This isolates a change in selected identities from a change in coverage, conditional on the reference rule; it does not match realized risk. Forced-count outputs may be worse than the baseline and are diagnostic, not deployment policies. Solver failures remain unmatched in the ledger, and zero-count matches do not count as coupling evidence. The completed v6 supplement evaluates this branch for every frozen seed, head and policy without selecting a deployment.

### 3.1 Explicit Costs and Joint Decision

For agent i, let s_i be the frozen past-derived scale and let L_i be mean
Euclidean error over the twelve requested future annotation steps divided by
s_i. The current implementation uses
s_i = max(observed path length, final causal speed times requested raw horizon,
0.001 dataset-local units). The numerical floor is not a verified physical
length. This transform can amplify errors after almost-stationary histories;
the primary metric remains fixed and native errors are reported separately.

Define b_i = max(L_i(B_i) - L_i(N_i), 0) and
h_i = max(L_i(N_i) - L_i(B_i), 0). Estimated net gain is
g_hat_i = b_hat_i - h_hat_i. OOF targets use held-physical-fold predictions,
whereas inference features contain only observed histories and frozen rollouts.
The estimated harm is constrained to be at least max(-g_hat_i, 0); that algebraic
coherence is not probability calibration or a realized-risk certificate.

For binary interventions a_i, the joint objective is

```text
min_a  -(1/n) sum_i a_i g_hat_i
       + lambda/|E| sum_(i,j in E) P_ij(a_i, a_j)

s.t.   a_i <= supported_i,
       sum_i a_i <= intervention_cap,
       (1/n) sum_i a_i h_hat_i <= predicted_harm_budget.
```

The pair term is zero when the graph has no edges. P is the nonnegative excess
of a normalized proximity proxy over the all-baseline pair; it is computed from
the two causal forecasts in a shared coordinate frame, not the future target.
The harm budget averages over all observed eligible agents, not only switched
agents. A policy cannot meet it merely by removing agents whose future labels
later prove incomplete. Pair penalties have no physical collision interpretation
without verified geometry and object extents.

The independent control sets the pair term to zero while retaining the same
support and budgets. Scene-uniform selection is deliberately restrictive under
a subunit intervention cap; if it cannot switch the entire scene, it reports
floor rather than changing the cap. Exact-count joint selection is a distinct
diagnostic conditioned on the independent count. Ordinary equal caps alone do
not isolate coordination. The all-baseline vector is feasible; failed or
unverified optimizer solutions return it. Estimated feasibility does not imply
that a learned intervention improves the realized trajectory.

### 3.1.1 Isolating Coupling From Single-Agent Geometry

The risk-only independent comparison does not isolate pairwise composition.
For binary choices and P(0,0)=0, the same penalty decomposes exactly as
P(1,0)*a_i + P(0,1)*a_j + d_ij*a_i*a_j, where
d_ij=P(1,1)-P(1,0)-P(0,1). A geometry-aware independent control retains the first
two terms and removes only the binary product. The signed d_ij is not clipped.
All controls use identical predictions and original risk/support constraints;
unary geometry and joint selection both match the risk-only reference count.
Geometry is not substituted for the estimated forecast-harm budget.

Writing the full objective as J=U+R, exact minimizers on the same feasible set
satisfy 0<=J(a_U)-J(a_J)<=R(a_U)-R(a_J)<=lambda*sum|d_ij|/|E|. This elementary
bound concerns the constructed objective only. It is not a trajectory-risk
guarantee or new theory. Additive pair penalties can favor the full policy over
risk-only selection even when R is identically zero. Therefore only comparisons
against the geometry-aware control can isolate the non-additive term in this
objective, and real forecast quality must still be evaluated separately.

The opt-in implementation passes160 constructed problems/320 enumerated optima
and99 past-input scene probes from33 existing source recordings. A legacy MILP
success flag proved insufficient on one query:6,435 feasible assignments show
a4.7883e-7 suboptimality. A versioned solver uses global positive objective
scaling and original-unit primal/dual/product checks. Old experiment-bound code
and results remain unchanged. The99 future-array poison checks pass, but no
future prediction errors are evaluated by these engineering probes. Three
constructed-score proxy gains are not evidence of neural or causal interaction
lift. See [full derivation and limits](interaction_controls_v1/method_and_limits.md)
and [completed numerical evidence](interaction_controls_v1/conclusions.md).

The [subsequent frozen-forecast study](frozen_interaction_v1/conclusions.md)
executes every completed v6 Transformer/EqMotion seed, head and fixed policy on
two already explored UCY development recordings of one physical site. Each of
24 combinations contains 970 scene queries and 37,775 agent queries, with 28,324
complete ADE labels. All original floor/candidate errors reproduce exactly;
these are new selection decisions, not newly trained predictors. Three new arms
share the checked numerical solver, and all match the reference count.

J-minus-U ADE is exactly zero in 21 combinations. The other contrasts are
-8.7911e-6 for Transformer seed29/ridge/conservative, and -1.5224e-5/+6.0173e-6
for EqMotion seed29/ridge/conservative/moderate. Transformer changes only one
ADE-labeled agent and recovers the risk-only outcome. All neural-cost-head
coupling contrasts are zero. EqMotion's effects change sign across the two
fixed policies. All 72 new control cells exceed 2% relative easy degradation.
The comparison therefore supplies no stable useful coupling effect. One physical
site cannot support a scene-level confidence interval; seeds and overlapping
windows are not substituted for independent sites. No parameter or policy is
selected from these post-hoc development outcomes.

The [frozen-risk diagnosis](frozen_risk_forensics_v1/conclusions.md) separates
predicted-budget feasibility from observed positive harm. Every predicted query
budget passes, yet 0-529 of 970 queries per fixed cell already exceed the realized
budget on available labels. The mean observed lower bound exceeds the mean
predicted harm in 63/72 cells. Missing selected costs remain unknown; depending
on the cell, 41-785 queries cannot be classified as within or exceeding budget.
This is descriptive evidence at one explored site, not an independent calibration
test or identification of the cause of underprediction.

There is also a target mismatch. Let d=L(N,Y)-L(B,Y), a denote intervention and
w=1{L(B,Y)<=tau} denote the outcome-defined easy group. The relative easy
requirement is sum(w*a*d)<=rho*sum(w*L(B,Y)), with rho=0.02 and a positive baseline
denominator. A global mean positive-harm cap constrains a different quantity.
Even the exactly scored within-budget query subgroups fail the relative easy
check in all 72 cells. Predicting E[w*max(d,0)|X] would address a joint conditional
moment that is absent from a purely global harm target, but fitting it would not
by itself supply a risk certificate. No revised model is trained or selected in
this diagnosis. Independent calibration support and the pending primary-metric
decision remain necessary before a new registered evaluation.

The subsequent [cost-head fit diagnosis](cost_head_fit_forensics_v1/conclusions.md)
shows that unreliable conditional costs already occur on the heads' fitting
rows. All12 fixed heads beat the fit-label mean reference on harm MSE, but all24
original per-agent eligibility subsets underpredict mean harm, and23 realize
negative mean net gain. The sole positive exception has44 fitting rows and is
not promoted. Nine heads overpredict harm globally, ruling out an exclusively
global mean-underprediction account of the observed failure. These eligibility
sets are not final scene-solver selections.

Ridge's zero projection affects10.58-50.75% of rows;65.37-74.33% of these still
incur positive harm. Neural softplus heads also show conditional failure, so
removing projection is not a remedy. The largest1% of harm labels contributes
56.23-98.72% of squared label mass, but this statistic does not isolate the cause
of the fitting error. Exact normalizer checks and54 original-batch replays find
no checked feature/target-order mismatch. OOF refers to the trajectory producer;
the cost head is evaluated in-sample. These findings refine the diagnosis, not
independent calibration, neural predictive utility or deployment safety.

The [validation-lineage audit](cost_validation_lineage_v1/conclusions.md) rules
out a simple reuse shortcut for the next head experiment. All18 original OOF
producers pass their own scene-exclusion checks, but the12 fitted cost heads
cannot be validated on those fitting rows. Moreover, all18 hypothetical
outer-fold head refits using the remaining OOF caches still have indirect
validation-scene exposure through the training producers. None of36 ordered
outer/inner exclusion requirements is met by the audited full/single-held-fold
producer pool. This is declared-lineage evidence, not a new accuracy result.
Nested upstream exclusion and train-only head preprocessing are needed for the
proposed held-cost comparison; independent calibration remains a separate data
requirement. The existing contract already rejects this exposure, and no
scientific split was changed or additional head trained during this audit.

The [conditional-risk derivation](conditional_decision_v1/risk_identities_and_counterexamples.md)
clarifies the inference: MSE regression is not intrinsically unsuitable. For a
true full-information conditional mean mu(X)=E[h|X] and a past-measurable policy
a(X), E[a*h]=E[a*mu]. For an approximate head and coverage c>0, selected mean
error can be as large as the global RMSE bound divided by sqrt(c). Score-only
calibration does not necessarily survive a policy using additional scene context.
For the easy constraint, E[w*h|X] also need not equal E[w|X]*E[h|X]. Four executable
finite examples establish these non-implications, not a learned improvement,
new theorem or confidence certificate. The aggregation and loss remain pending
scientific decisions; no replacement target has been fitted.

### 3.2 Restricted Action-Class Diagnostic

Before learning another gate, consider the finite pool of frozen forecasts N_k
and actions B + alpha*(N_k-B), with one alpha in [0,1] for each whole agent path.
For nonnegative fixed scene weights w_i, define the label-side diagnostic

```text
ell_i_star = min over k, alpha of L_i(B_i + alpha*(N_ki-B_i), Y_i)
oracle_gain = 1 - sum_i w_i*ell_i_star / sum_i w_i*L_i(B_i,Y_i).
```

Any joint policy confined to these actions has labeled-set loss at least
sum_i w_i*ell_i_star. This follows directly from the rowwise minimum; adding
compatibility constraints cannot enlarge its feasible action set. It is an
elementary diagnostic, not a new statistical theorem. The target enters only
this evaluator and is never available to a causal policy.

For fixed k, mean Euclidean error is convex in alpha. Subgradient bisection and
a mean-correction-norm Lipschitz envelope numerically bracket the minimum.
The implementation includes baseline/candidate endpoints and separately checks
its solutions with a scalar optimizer. This does not bound arbitrary mixtures
of several candidate vectors, per-waypoint scaling or newly trained predictors.
It also gives no guarantee about population risk under scene shift.

## 4. Experiments and Remaining Evidence

### Training Exposure and Candidate Ceiling

A matched sampling study retains all 11,966 approved fit windows, three seeds,
three physical-scene folds, the same MLP and 4,000 updates. Fifty-four new fits
compare scene, track and event/track balance against eighteen exactly replayed
row-uniform controls. Directed image-motion primary gains versus CV are -0.9724%,
-1.2393%, -1.1877% and -10.4303% respectively. No fresh fit meets both primary
improvement and easy preservation. Raising the frequency of scarce supervised
events does not add independent observations or transferable onset direction.
[All conditions and failures](track_event_sampling/conclusions.md).

The subsequent diagnostic uses all frozen forecasts, without retraining or
opening development/calibration/confirmation roles. The pool's perfect binary
oracle is 1.62653%; its whole-path scaling oracle is 1.72618%. Seed gains for the
latter are 1.57542%, 2.14971% and 1.45339%. A 2,000-draw descriptive resampling
interval over only three exposed fit scenes is [0.73625%, 3.06827%]; it is not a
confidence interval for a learned policy. All 72 oracle computations reproduce
exactly, and 864 sampled independent scalar optimizations agree to 1.777e-15.

Exactly-static histories are 365/11,966 rows and contribute 89.3667% of equal-scene
normalized CV error. For the 188 static-to-movement rows, the pooled oracle gain
is only 0.19903--0.43404% across seeds. Conditional moving-history headroom is
larger, but those rows cannot replace the registered complete-cohort primary.
The numerical scale floor amplifies static errors; this finding does not alone
explain the earlier negative native-coordinate diagnostics. No oracle is treated
as an inference result. [Restricted ceiling and limitations](candidate_headroom/conclusions.md).

### Output Range Versus Supervision

We next separate linear versus numerically capped sinh residual readouts from
log1p-ADE versus asinh-residual SmoothL1 supervision. The same features, seeds,
row sampler and update budget yield 54 fresh fits and 18 exactly replayed linear
controls. This training surrogate does not change the primary evaluation metric.
Directed-feature held gains are -0.97244%, -0.90994%, -15.79533% and -171.03448%
for linear/log, sinh/log, linear/asinh and sinh/asinh respectively. All new fits
fail positive complete-cohort gain plus easy preservation.

Transformed supervision raises the mean overlapping fit-cohort gain diagnostic
from 1.17% to 11.69% (linear) or 19.68% (sinh), but held static-stay errors become
large despite a zero CV floor. Native-coordinate per-recording diagnostics also
remain negative. Better in-sample fitting therefore does not identify transferable
onset/direction cues. The numerical cap is not a physical safety constraint.
All 72 checkpoint inferences replay exactly; a completed resume leaves 145
artifact hashes unchanged. [Design, complete results and failure slices](residual_range/conclusions.md).

### Past-Only Coordinate Conditioning

A further matched experiment separates past-only frame conditioning from a
no-anchor CV guard. Thirty-six new fits and eighteen replayed controls keep the
features, dimensions, objective, seeds, folds and 4,000-update budget fixed.
The frame uses past ego velocity, supported neighbor motion or relative neighbor
position; it reads no future labels. Eight anchorless rows retain exact CV.

Past-frame primary gains versus CV remain negative: -0.86256% with quality
controls and -0.88781% with directed motion. All new fits fail positive primary
gain and easy preservation. Measured prediction disagreement under quarter-turn
rotations vanishes on static histories, but this does not improve forecasting
enough to beat CV. Original moving source inputs were already heading-aligned;
stress-testing rotated, already-aligned features is not a source-pipeline
equivariance test. The static-history result is the relevant narrower diagnostic.

All 54 checkpoints replay exactly; completed resume preserves 109 hashes without
updates, and 24 focused tests pass. These are exploratory fit-scene comparisons,
not independent confirmation. Coordinate consistency does not supply missing
launch intent or justify an architectural novelty claim.
[Design, scope and failures](past_frame/conclusions.md).

### Start Information Versus Displacement Prediction

A separate fit-only probe tests whether the observed motion summaries can predict
any recorded future coordinate change after a static history. Four nested inputs,
two fixed classifiers, three seeds and two supported scene folds give 48 fits.
No trajectory model is trained. All 365 stationary windows remain; they represent
only 31 local IDs and 45 runs, not independent samples or new confirmation sites.

With ExtraTrees, magnitude/directed inputs give Hotel-to-ETH AUROC 0.8105/0.8215
and absolute Brier lift 0.08030/0.07870 over the training-only prior. Reverse
transfer gives AUROC 0.5004/0.5118 and negative mean Brier lift. Every logistic
variant is worse than the prior. For the tree models, added motion versus
quality-only has uncertain agent-balanced lift, with intervals crossing zero. These conditional held-agent
intervals do not establish new-scene generalization.

The localized signal argues against declaring motion wholly uninformative, but
neither bidirectional probability transfer nor departure trajectory accuracy is
established. No residual or switching policy is promoted. All 48 classifier
predictions replay exactly. [Full outcomes and limitations](motion_start_information/conclusions.md).

### Auxiliary Video Integrity, Not Forecast Evidence

Before prospective auxiliary training, a full local SDD source audit decodes
522,497frames and checks10,616,256annotation rows.54of60reference/video pairs
have unequal image dimensions. Nexus video naming also follows a lexical
reindexing pattern rather than annotation identity; a fixed source-link repair
changes10media paths and resolves three frame-range mismatches. No raw file or
trajectory label is changed, and no model is fitted in this audit.

The resulting diagnostic source map is not a training registration, independent
confirmation or metric/time calibration. Missing/padded image support and
offline interpolation provenance remain material. These defects concern local
SDD media, not a post-hoc explanation of the existing ETH/UCY negative forecasts.
[Source audit and repair](sdd_media_alignment/conclusions.md).

### SDD Geometry-Image Join and Its Limits

The [past-only geometry bridge](sdd_step_bridge/conclusions.md) uses the original
40 SDD train recordings, comprising 8,005,367 annotation rows in five scene
folders. It indexes 3,045,974 eight-point histories at raw stride 1 and 229,333
at raw stride 12. These are overlapping input windows, not independent samples.
Twelve requested forecast points correspond to offsets 1,...,12 or
12,24,...,144 raw frames respectively. Neither is asserted to match ETH/UCY
elapsed time, and neither is the supplementary raw-frame t+50 task.
Index membership depends on past support only; incomplete future labels remain
separately masked rather than removing an input or becoming a zero-error result.

The [multimodal join](sdd_multimodal_bridge/conclusions.md) connects a fixed
input-selected subset of 5,074 geometry queries to 39,144 unique frame/agent
image crops. Of these, 38,449 occur after frame 63, beyond the old prefix-only
image cache. Some retained pixels exist at every step of each sampled history,
but 3,721 crops are geometrically partial and 84 intersect inferred dark-border
support. An independent sampled decode reproduces 143 crops from 120 frames;
it does not certify all crops, actor visibility or semantic identity.

The projected short side of 15,145 annotation boxes (38.69% of crop requests)
is below eight pixels after pooling. This descriptive property is not a
visible-body measure, an exclusion rule or proof that resolution caused the
earlier ETH/UCY failures. The image cache serves the registered diagnostic
queries, not all 229,333 stride-12 histories. There are zero optimizer updates
and no predictive evaluation in this bridge. The subsequent separately registered
comparison below expands the image cache and supplies actual training evidence;
that result is not inferred from joins, source replay or an untrained forward.

### Matched SDD Auxiliary Training

Following explicit source approval, the [registered source comparison](sdd_auxiliary_v1_decision.md)
builds a complete past-image cache for all 229,333 eligible stride-12 windows
from the original 40 training recordings. It retains 188,358 complete, 37,360
partial and 3,615 absent future-label windows without future-based membership
filtering. Masked losses distinguish absent labels from zero error. Original
SDD validation/test raw inputs remain unused. Each source fit samples 128,000
draws with replacement, covering 97,892-98,207 unique windows, not a full epoch.

The 2 x 3 source/modality design uses three seeds and three physical-site folds:
54 fresh fits, 324,000 optimizer updates and 10,660.60 summed fit seconds.
Both schedules use 2,000 first-phase and 4,000 main-task updates; the first phase
is either main-fold training or SDD supervision. We retain weights, reset the
optimizer/main sampling stream at the boundary, and use the same main-train-only
normalization. Every final checkpoint is evaluated without held-score selection.
The existing small MLP/CNN architecture is unchanged; this is not new latent
generative, JEPA or Transformer training.

| Input | No-SDD gain vs CV | SDD gain vs CV | SDD vs matched neural control |
| --- | ---: | ---: | ---: |
| Geometry | -1.34976% | -0.80523% | +0.53728% |
| Coverage masks | -1.37243% | -0.81626% | +0.54864% |
| Past RGB | -1.98266% | -1.27335% | +0.69552% |

These are gains in the fixed equal-scene past-normalized ADE. All 54 held fits
remain negative and fail easy preservation, despite positive fit-cohort gains
of 0.56870-2.17476%. SDD reduces degradation versus neural controls, not versus
CV. SDD RGB versus its mask control is -0.45339%, with a descriptive 2,000-draw
scene-bootstrap interval [-0.63261%, +0.49055%]; its relative benefit occurs only
on Zara. There is no stable pixel contribution or deployable candidate.

Easy relative degradation is retained with absolute normalized harm, because
near-zero baseline errors amplify ratios. Static-stays CV error is exactly zero,
so its percentage gain is undefined; absolute model error is reported instead.
All checkpoints replay exactly and completed resume preserves 163 artifacts with
zero updates. The analysis-only zero-reference fix changes no training or metric.

The source contrast also changes main-domain exposure: 6,000 main updates versus
2,000 source plus 4,000 main. It cannot isolate transferable representation from
regularization or reduced main-domain overfitting. Source feature clipping,
image detail, supervision support and horizon mismatch remain possible factors,
not identified causes. Three repeatedly exposed sites, even with seeds and
bootstrap, cannot establish independent transfer. [Complete results](sdd_auxiliary_v1/conclusions.md)
and [failure analysis](sdd_auxiliary_v1/failure_analysis.md) retain these boundaries.

### Source Supervision Versus Main-Task Exposure

The [registered mechanism comparison](sdd_auxiliary_mechanism_v1/report.md)
adds 54 fresh fits (270,000 updates) and reuses the previous 54 fits only as
hash-verified controls. A main4k arm starts randomly and matches the main-phase
sample stream. A permuted-source arm matches source/main draws and compute,
but replaces source correction labels with donors from the same recording and
exact future-support stratum. Inputs and population membership are unchanged.
Singletons and same-agent donors are retained and disclosed; this negative
control does not remove all recording-level information or dependence.

| Input | Main4k vs CV | Permuted source vs CV | Real vs permuted source |
| --- | ---: | ---: | ---: |
| Geometry | -1.03110% | -0.92421% | +0.11789% |
| Coverage masks | -1.00276% | -0.80078% | -0.01536% |
| Past RGB | -1.52648% | -1.33240% | +0.05827% |

Main4k outperforms main6k in all seed-averaged input/site comparisons. Thus the
previous auxiliary advantage partly reflects main-training exposure. Real versus
permuted source intervals are [-0.12649%, +1.89699%], [-0.15366%, +0.15464%] and
[-0.17034%, +0.27215%]; correct source pairing has no stable demonstrated benefit
in this design. This is not an equivalence test. Every schedule/input aggregate
still loses to CV. Three new individual fits have tiny positive gains but fail
easy preservation; there are zero safe positive fits across the 108-fit matrix.
New-fit absolute easy harm is 0.02593-0.29810 normalized ADE. All new checkpoints
replay exactly and completed resume preserves 166 artifacts without updates.

A train-only audit also identifies a stationary scaling mismatch. The fixed
0.001 native-unit floor does not preserve normalized stationary targets across
coordinate units. Broad static-moves target medians are 1,125 in SDD and 24.702
in main fold-0 training data; analytic scalar log-loss sensitivity differs
accordingly. These are not neural parameter-gradient measurements, physical
movement comparisons or identified causes. Broad source static-moves support
(10,039 windows,342tracks) differs from the earlier half-box proxy (244windows,
58tracks in train40). Source semantics, unit-invariant internal representation
and loss response require controlled follow-up, not a new success claim.

The main estimand and roles remain fixed. All site intervals are descriptive
over three previously exposed sites, and no model is selected from these held
scores. These controls narrow the source-transfer interpretation without
establishing deployable forecasting or independent generalization.

### What the Current Evidence Can Establish

The [observed-unit comparison](unit_frame_training_v1/report.md) adds 27 fresh
geometry fits with nine verified controls, preserving all populations, source
schedule and the primary metric. Gradient probes use only training roles;
three exposed site folds evaluate the fixed matrix without model selection.

| Conditioning | Equal-site primary gain vs CV | Safe positive fits |
| --- | ---: | ---: |
| Verified old source-trained geometry | -0.80523% | 0/9 |
| Reconstructed unit inputs | -0.68730% | 0/9 |
| Unit inputs and radius decoder | -2.48928% | 0/9 |
| Radius decoder and internal log loss | -185.77715% | 0/9 |

The input-only contrast against legacy is +0.11699%, with a descriptive site
interval [-2.07626%, +0.59275%]. All three Hotel seeds are positive but unsafe;
Zara worsens. Under radius decoding and the original objective, every logged
source batch and 99.44% of logged main batches are gradient-clipped. Internal
loss removes logged clipping yet strongly increases primary error and easy harm.
For past radius R and primary error e, its derivative with respect to e is
1/(R+e): improved numerical conditioning is not equivalent to optimizing primary
forecast accuracy. The tested repair is insufficient, not a general impossibility.

A future-label oracle over CV plus all four geometry variants reaches only
2.63722% aggregate gain. This is a post hoc diagnostic upper bound for this
finite pool, not learned selection or independent evidence. The 27 checkpoints
replay exactly and completed resume preserves 82 artifacts. No multimodal
or deployment claim is made by this conditioning experiment.

The [source-supported start probe](source_start_probe_v1/conclusions.md) removes
trajectory regression and tests a binary annotation-change objective on476
past-only unit-frame features. It adds22,374 complete-label stationary SDD windows
from726 local agent IDs, without opening source val/test or the main sealed roles.
Forty-five fresh logistic/tree/MLP fits produce54 prediction cells; source-only
models are shared across directions, not fitted twice. MLPs receive15,000 total
updates. Main remains365 exposed stationary windows from31IDs/two sites.

Source-only Brier lifts versus the main training prior are-0.07197/+0.06887
(ETH/Hotel) for logistic,-0.11032/+0.06676 for trees,and-0.11919/+0.05899 for MLP.
Mixed training also worsens ETH in every family. No arm is positive in both
directions. The source positive fraction44.87% is close to Hotel45.42%,while the
opposite-main-site reference predicts72.29%. Constant source-prior Hotel
Brier0.24794 beats every source/mixed family. The exact diagnostic decomposition
`(q-r)^2-(mean(p)-r)^2+2*Cov(p,y)-Var(p)` attributes the positive Hotel contrast
to mean-probability shift; the varying-prediction term is negative on average in
both sites for every source/mixed family. This is not proof of zero information
or a calibrated inference-time correction. It prevents interpreting the partial
gain as demonstrated sample-specific switching capability.

The2,000-resample conditional agent intervals are descriptive,with only five
ETH and26HotelIDs. All Hotel source/mixed intervals versus the prior cross zero
under agent weighting;row and agent estimands are not interchangeable. Forty-five
probability replays are exact;completed resume preserves166 artifacts. No model
is promoted. Main-only overfit,limited target support,annotation semantics and
unverified physical-time correspondence remain alternatives to an intrinsic
absence of predictive information.

The subsequent [matched visual probe](source_visual_start_v1/conclusions.md)
adds 30 real Torch fits and 60,000 updates with paired RGB/coverage-only sampling.
It preserves the same stationary cohort and labels. The mixed schedule gives
positive window-weighted RGB-minus-mask Brier lifts of 0.039378 on ETH and
0.036045 on Hotel, positive in all three seeds. This primary contrast is retained.
However, equal-agent contrasts are -0.019117 and -0.038780, with conditional
intervals [-0.133658,0.083091] and [-0.149999,0.062546]. Both mixed RGB models are
also slightly worse than their own constant training priors. SDD-only RGB worsens
ETH versus mask by 0.090782 Brier. No schedule beats its own prior in both sites.

The score decomposition distinguishes partial improvement over a weak variable
predictor from useful probability estimation. Mixed ETH's row gain consists of
0.079563 from mean-probability shift and -0.040185 from the varying-prediction
term. Mixed Hotel improves the latter term relative to mask, but not enough to
beat its own constant prior. Its equal-agent contrast remains negative after
each single-agent omission. These are descriptive diagnostics, not selected
calibration corrections or independent scene evidence.

All 30 probability replays are exact and completed resume preserves 91 artifacts.
Past crops exist and change over time, but source annotation boxes occupy a
median of approximately 11 by 13 model pixels; body-state visibility and shared
behavioral label semantics remain unverified. This finite-budget probability
probe establishes neither trajectory utility nor the absence of visual
information in general. No forecast or deployment is promoted.

The [source-internal held-site control](source_site_probe_v1/conclusions.md)
then keeps that architecture, cohort and supervision fixed while fitting on four
SDD sites and holding the fifth out. All 30 fits complete: two paired RGB/mask
arms, five physical sites and three seeds. Equal-site RGB-minus-mask Brier lift
is -0.020046, with a conditional 2,000-resample site interval
[-0.027587, -0.011995]. All five site means and 14 of 15 seed-site contrasts are
negative. Both arms lose to their training constant priors in all 30 fits.
RGB lowers training Brier (0.202032 versus 0.211919) while raising held Brier
(0.297592 versus 0.277546). Equal-agent weighting remains negative in aggregate.
Thus cross-dataset mismatch alone is not an adequate explanation for this
configuration's failure. Five exposed sites and overlapping training folds do
not provide independent confirmation or a proof that RGB cannot help.

A source annotation audit further limits interpretation of the classifier.
The median maximum displacement among 10,039 positive annotation-change labels
is 2.06 pixels; 8,583 are below one tenth of the current box diagonal. The 244
half-diagonal-or-larger windows constitute 1.091% of stationary windows but
26.155% of stationary CV ADE error mass. Smaller changes still account for
43.406% of that mass. Binary event counts therefore cannot replace a
forecast-cost objective, nor can smaller events simply be discarded. These
source-only descriptive bins do not change the main endpoint or train labels.
Offline generated annotation rows are not human intention labels. Thirty exact
replays and 91 unchanged completed-resume artifacts verify execution, not utility.

The [source trajectory-cost experiment](source_cost_dynamics_v1/conclusions.md)
then tests twelve-step bounded forecasts directly rather than another binary
target. Its fixed two-by-two comparison (ADE/log-ADE and RGB/mask), five source
sites and three seeds comprises 60 new fits and120,000 updates. Initial outputs
exactly reproduce stationary CV; no-context rows remain unchanged. Equal-site
uncontrolled ADE gains are -1.66534%/-1.69835% for ADE mask/RGB and
-1.60683%/-1.56871% for log-ADE mask/RGB. All four conditional site intervals
are negative. Both matched RGB contrasts have intervals crossing zero. All60
held-site fits also fail under the fixed0.9 classifier guard; reducing the
intervention rate reduces damage without producing positive gain.

The failure differs from the preceding classifier overfit: every complete
training-set ADE also worsens CV, by1.07-2.05%. Reconstructing exact training
draws removes minibatch-difficulty confounding in the loss trace. Initial excess
ADE is zero in all60 models; final logged excess is positive in all60. Logged
gradients are clipped in100% of ADE and98.41% of log-ADE batches. This motivates
training-only optimization and output-scale diagnostics, not a claim that
clipping is the established cause or that no predictive information exists.
The source containing-ball oracle retains99.67-100% headroom by site, whereas
actual learned candidates add little useful oracle headroom. The binary and
geometric oracles use future labels diagnostically and are never inference inputs.

Easy samples have zero CV error and acquire0.02173-0.02392 native-pixel mean
absolute harm across uncontrolled arms; a percentage easy-degradation gate is
undefined. Context bounds are not physical-safety certificates, and the frozen
guard is not independent risk calibration. Sixty exact prediction replays,
15 matched four-way training streams and181 unchanged completed-resume artifacts
verify execution. Five repeatedly exposed source sites, overlapping folds and
offline silver histories still do not supply main-task independent confirmation.
No predictor, threshold or deployment is selected from this negative matrix.

The [training-only microfit control](source_microfit_v1/conclusions.md) narrows
the implementation diagnosis without adding a benchmark claim. It selects16
feasible nonzero training targets and adds16zero targets in a second cohort,
using distinct scoped agents from the source training complement. Two decoder
laws and three seeds give12fits/24000updates. Both original and rescaled models
memorize these rows: mean training ADE reductions are98.18%/98.44% on nonzero
targets and96.60%/96.99% on the mixed cohort. Original gradients are clipped at
every update, yet fitting succeeds; rescaling is not better in every seed.

This refutes a universally disconnected learning path, not an absence of
full-corpus optimization problems. Every microfit row receives2000passes, while
the original matching15430-row training complement receives8.2955draws per row
on average. Exposure, diversity and gradient noise differ. Training-label
feasibility selection cannot become a held-set filter; zero-target absolute
harm also remains positive. Twelve exact replays and37unchanged completed-resume
artifacts establish reproduction only. A controlled full-training learning
curve is required before attributing the earlier failure to insufficient
exposure, step size or conditional-information limits. No held forecast is
computed or deployment upgraded by the microfit.

The [full-training continuation](source_continuation_v1/conclusions.md) then
keeps all15,430 source-training rows outside bookstore and forks three verified
step2000 parents into constant-rate and cosine-decay branches. Six branches add
48,000 updates, with fixed full-training evaluations at2000/4000/6000/10000.
At the final endpoint, constant continuation still loses to CV by1.0310% on
average; cosine yields+0.2533% training gain, positive in all three seeds
(range+0.2039% to+0.3259%). This is not a held-scene result or confidence interval.
Decay reduces zero-target absolute harm from0.046223 to0.012653 annotation pixels
relative to the constant schedule, while reducing moving-target benefit from
1.2446% to0.8762%. It changes the gain/harm balance, not simply movement accuracy.

All24 training predictions replay exactly and three paired sample streams match.
All new updates still clip gradients; the schedule, including its effect on
AdamW shrinkage, partially repairs training fit without proving a unique cause.
Mean exposure is41.48 draws per row, not evidence of convergence. Zero-target
harm remains positive and relative easy degradation undefined. No held-source
or main evaluation, model selection or deployment occurs in this diagnostic.
The next required evidence is conditional forecast utility outside the fitting
rows, including a matched visual-input control; the method contribution remains
unproved. The previously reported negative comparisons are not overwritten.

The [matched modality follow-up](source_transfer_control_v1/conclusions.md)
adds six coverage-only continuations with 48,000 new updates, preserving the
same training rows, parameter count, optimizer and sampled streams. All six RGB
endpoints and six parent states are reused after verification. All 18 fixed
predictor states are scored on the already-explored bookstore source site,
without model or threshold selection. Coverage-only cosine training also gains
0.1872%, but its held-source gain is -1.7438%; RGB cosine yields -2.0806%.
The paired RGB-minus-mask contrast is -0.3368 percentage points, with conditional
recording-block interval [-1.0851, -0.1669]. Both decay-versus-parent intervals
cross zero. Thus the small training repair does not establish transferable
visual utility. All 18 states remain negative with the fixed classifier guard.

All 42 predictions replay exactly and 202 artifacts survive completed resume
and repeated scoring unchanged. The 2,000 bootstrap draws concern seven
recordings of one previously explored physical site, not independent scene
confirmation. Easy percentage degradation is undefined at zero baseline error;
absolute harm stays positive. A supplementary post-hoc complete-path oracle over
the 18 forecasts plus CV offers only 1.2752% aggregate gain and 0.5267% hard gain.
This is an empirical limit for selecting among those paths, not for new models,
path blending or other populations. It motivates candidate-quality repair,
not another retrospective threshold search. No deployment is changed.

The subsequent [training-only cost-deferral repair](source_cost_deferral_v1/conclusions.md)
adds an exact stationary-baseline action, holding the observed schema, sampled
streams and cosine update budget fixed. Six fresh continuations add 48,000
updates; three dense controls are verified and reused. A plain expected-action
cost collapses to full rejection in all three seeds. A fixed package of
signed-gain supervision and an additional proposal-loss term gives mean gated
training gain 0.233917% versus 0.187244% for the dense control. Every seed has
positive paired aggregate gain; the mean difference is only 0.046673 percentage
points, or 0.00052631 native annotation pixels.

This is a cost-allocation tradeoff, not improved dynamics. The raw proposal
gains 0.184359%, slightly below dense control. Gating reduces zero-target harm
from 0.01077451 to 0.00699713 pixels, but also reduces hard gain from 0.325144%
to 0.276383%. deathCircle remains negative in every seed. Relative easy damage
is undefined at zero CV error, not a passed safety gate. All 24 snapshots replay
exactly and 290 artifacts survive completed resume unchanged. No held-source
or main evaluation is performed, and seed ranges are not generalization CIs.
An explicitly post-hoc score diagnosis finds in-sample gain fitting better than
a constant-mean target, but the global Huber/mean discrepancy is small and does
not explain the plain objective's rejection. Independent calibration and useful
scene-level joint intervention remain unestablished; no model is promoted.

The [separately registered fixed source readout](source_deferral_transfer_v1/conclusions.md)
now evaluates all six final cost-deferral endpoints and the three matched dense
controls, without additional training or threshold selection. Cost-supervised
hard actions yield -1.245558% gain versus CV, conditional recording interval
[-4.430101%, -0.566922%], compared with dense -1.743843%. The paired advantage
over dense is +0.498285pp, interval [+0.242032, +1.438712], but every seed still
loses to CV. The raw proposal does not improve the dense decoder, and the plain
expected-cost head outputs only CV. Thus the training improvement is not a
positive forecasting transfer result.

The intervention rate is 62.985%, compared with 61.002% on training data.
Hard gain is -0.036330%; equal-recording and equal-agent gains remain negative.
Zero-target absolute harm is 0.01581748 annotation pixels, with undefined
percentage degradation. All nine predictors replay exactly, and repeated
scoring preserves 313 artifacts. The 2,000 paired recording resamples remain
conditional on seven recordings at a single historically explored site; no
independent confirmation is implied.

An additional post-hoc training-only exact-input audit finds 153 duplicate-input
rows among 15,430, including 38 rows with conflicting target trajectories.
The tested sufficient zero-optimum condition covers none of the conflicting
groups. This does not support widespread exact input aliasing as the main
failure mechanism, but does not establish sufficient causal information or
rule out approximate ambiguity. Optimistic in-sample candidate-cost targets
and moving-target optimization remain hypotheses requiring training-side
cross-fitting or frozen-candidate controls, not another held threshold search.

The subsequent [training-side candidate cross-fit experiment](source_crossfit_v1/conclusions.md)
trains twelve cold-start models, four inner source-site folds and three seeds,
for 120,000 updates. Each producer and its fitted preprocessing exclude the
query's site and bookstore. All twelve training-complement gains are positive,
but all twelve inner-held gains are negative. Primary equal-site gain is
-5.01598%, conditional four-site interval [-8.39655%, -2.48773%]; the
window-weighted sensitivity is -5.45692%. These are explored training sites
with shared fold fits, not independent confirmation. The comparison with the
cached full-four-site in-sample reference changes training size, site exposure
and normalization, so it does not isolate training optimism.

Fixed candidate/CV oracle gain is only 0.52707% window-weighted. A separately
marked post-hoc attribution gives equal-site oracle 0.46765% and partitions the
5.01598pp primary excess into 4.32112pp on zero-target rows and 0.69486pp on
nonzero-target rows. The latter subset still loses 0.93253% in the window view.
Thus invented movement explains most measured harm, while simple perfect
easy-case rejection would not repair the remaining candidate's average error.
This is a limitation of the tested fixed action family, not a general
learnability bound. Twelve exact prediction replays and zero-update resume
verify implementation, not dynamics utility. No new risk head is trained;
future risk-head validation must also exclude its validation site from upstream
training-label producers. No outer/main scores or deployment are added.

The [matched static-gradient control](source_motion_candidate_v1/conclusions.md)
then adds twelve cold-start fits and 120,000 updates, retaining full sampler
exposure while removing zero-target ADE gradients only. Actual equal-site gain
drops to -98.71920%, conditional interval [-128.49923%, -66.02162%]. Nonzero-target
gain is also negative (-32.38720% window-weighted), and zero-target harm grows
from 0.09190 to 1.42880 annotation pixels. Thus static-target gradients suppress
large candidate movement, but removing them is not a useful predictive repair.
The matched future-oracle increase is +3.29204pp, interval [+3.04645, +3.64157],
reaching 3.75968%; it does not demonstrate causal switchability.

A separately registered post-hoc direction control preserves candidate path
shape and magnitude while rotating it by +90, -90 or 180 degrees. Its oracle
gains remain 3.49561%, 3.58218% and 3.31027%. Original-minus-null contrasts are
+0.26407, +0.17750 and +0.44941pp, with the -90 interval including zero. These
unadjusted explored-site contrasts caution against equating oracle headroom
with accurate direction; they do not establish direction irrelevance or
independent generalization. All twelve models replay exactly. No new risk head,
independent calibration, main evaluation or deployment is added.

The subsequent [raw annotation and past-box audit](source_motion_quality_v1/conclusions.md)
retains all 15,430 queries. Among 6,864 nonzero futures, 46.63% stay within two
annotation pixels and 82.07% within five. Only 207 queries reach half the median
observed box diagonal. However, 728 queries above ten pixels account for 53.25%
of summed normalized baseline ADE and both candidate families still lose there.
Small annotation changes therefore do not explain away the prediction failure.
No bin is removed or called human-verified motion or noise.

Forty-eight fixed ExtraTrees probes compare geometry against geometry plus
38 past-box shape features across four held sites and three seeds. Equal-site
Brier lifts versus training prevalence are -0.001959/-0.023872 for any future
change and -0.0001240/-0.0001652 for half-box excursion. The added-box contrast
for any change is -0.021913, conditional four-site interval [-0.041205,-0.002622];
the half-box contrast interval crosses zero. This tested feature addition does
not repair probability transfer. Changing feature count also changes forest
random subspaces, and rare half-box labels limit power; neither result proves
that all appearance information is useless. Exact replay succeeds for 48 models.

Raw provenance also shows that 15,316 histories include generated annotations
whose next source control is after the query. The experiment remains an offline
annotation forecast, not certified sensor-as-of perception. Raw future-box and
loaded-target mutation checks verify the implemented feature boundary, not how
the dataset annotations were originally constructed. All intervals are
conditional on four previously explored sites with shared fitting populations;
no independent confirmation or new trajectory gain is established.

| Question | Observed result | Supported conclusion |
| --- | --- | --- |
| Can routing rescue the frozen candidate family? | Oracle gains 1.62653%, or 1.72618% with whole-path scaling | Limited labeled-set headroom for this action class, not a global impossibility result |
| Does repairing coordinates suffice? | Past-frame gains -0.86256% / -0.88781% versus CV | Tested consistency repair is insufficient for useful forecasting |
| Do motion features transfer start information? | Hotel-to-ETH positive Brier lift; reverse negative; added-motion intervals cross zero | Localized probability signal, no stable bidirectional or trajectory contribution |
| Are additional SDD past modalities available? | Full auxiliary cache of 229,333 windows and 254,841 past crops | Input access established; visibility and predictive benefit require separate evidence |
| Does SDD supervision fix transfer? | 54 matched fresh fits; source helps neural controls, but 0/54 safe positive fits | Tested auxiliary schedule insufficient; no deployment or independent confirmation |
| Is the advantage specific to correct source pairing? | 54 new controls, 54 cached fits; all real-versus-permuted intervals cross zero | Main exposure accounts for part of the old difference; stable conditional transfer remains unproved |
| Does unit conditioning repair prediction? | 27 fresh geometry fits; input-only -0.68730%, internal-loss -185.77715% vs CV; 0/27 safe | Engineering repair and balanced gradients are insufficient for protected forecasting |
| Does source supervision transfer start information? | 45 classifiers;no bidirectional gain;Hotel source prior beats every source/mixed arm | Probability-level source benefit remains confounded by prevalence,not demonstrated forecast gain |
| Does past RGB add robust start information? | 30 matched fits; mixed window gain positive but agent-weighted gain negative and own-prior comparison fails | Fragile partial contrast, not robust transfer or trajectory utility |
| Does the same RGB representation generalize inside SDD? | 30 matched fits; all five held-site mean contrasts negative; equal-site Brier lift -0.020046 | Failure is not only source-to-main transfer; supervision and visible-event support need controlled repair |
| Does direct trajectory-cost training resolve the problem? | 60 fits; uncontrolled ADE gains -1.57% to -1.70%; training ADE also worse in every fit | The tested cost repair is insufficient; optimization and candidate trajectory utility remain unresolved |
| Is the neural fitting path universally broken? |12training-only microfits learn selected16/32-row cohorts; original decoder also succeeds with100%gradient clipping | Numerical fitting is possible; not a benchmark repair or proof of useful predictive information |
| Does longer exposure or rate decay improve complete training fit? |6continuation branches/48knewupdates;constant -1.0310%,cosine +0.2533% | Small schedule-dependent training repair; no held gain or easy-preservation proof |
| Does that repair transfer, and does RGB help? | Six matched mask continuations; cosine held gain mask -1.7438%, RGB -2.0806%; all 18 states negative | Training improvement does not establish held utility; paired RGB contrast negative, no deployment |
| Does exact-baseline cost deferral repair training? | Six new fits; plain objective all-reject; supervised cost gives +0.2339% training gain vs dense +0.1872% | Small routing tradeoff, weaker hard benefit, no stronger decoder or held evidence |
| Does that deferral benefit survive the fixed source readout? | All three supervised-cost action seeds lose to CV; mean -1.2456%, versus dense -1.7438%; plain action all-rejects | Reduces neural harm without positive transfer; current candidate decoder and cost transfer remain insufficient |
| Is large-scale exact observed-input conflict the main supported cause? | 153 duplicate-input training rows, 38 conflicting rows, no sufficient zero-optimal conflicting group | Not supported by this exact-schema diagnostic; approximate ambiguity and feature sufficiency remain open |
| Does producer-excluded candidate prediction yield useful cost supervision? | Twelve cold-start fits; equal-site OOF gain -5.016%; fixed binary oracle below 0.53% | Provenance-correct costs are available, but tested candidate utility remains insufficient for a large routing gain |
| Is baseline-relative joint intervention validated? | No stable advantage in the matched-count predictor study | Main methodological contribution remains unestablished |

These rows summarize different experiments and estimands; their scores must not
be pooled into a single success rate. In particular, probability ranking,
coordinate consistency and correct data plumbing cannot replace forecast gains
with easy-case preservation. Reusing controls also means fit counts across
reports cannot simply be summed as independent experiments.

### Frozen Pretrained Appearance and Temporal Readout

A source-only matched experiment tests frozen ImageNet ResNet18 features with
geometry-only, current-image and eight-frame appearance controls. All 36 heads
complete 10,000 updates with the same all-target ADE objective, full-population
sampling and four-site/three-seed design. No encoder fine-tuning or held-based
selection occurs. The population is the same 15,430 stationary-history source
queries, not the complete main benchmark or independent confirmation set.

Equal-site gains versus stationary CV are -0.0704%, -1.9079% and -6.1022%; all
36 held fits are negative. Sequence-minus-current is -4.1943 percentage points,
conditional site interval [-6.2288,-2.6727]. Sequence training gains are positive
but do not transfer. Zero targets account for 90.96% of window-weighted excess
error, while nonzero-target predictions also lose 0.5883%. Easy percentage
degradation is undefined against a zero-error floor; absolute pixel harm is
retained. Each fixed arm's binary oracle is a diagnostic, not a learned policy.

All heads replay exactly. Three encoder batches replay from raw historical crops;
all 25,300 image rows and 15,430 query mappings align. Completed resume adds zero
updates. This is negative evidence for the tested pretrained readout, not against
all visual dynamics. Source crops remain 32 x 32 before upsampling; offline label
interpolation, four explored sites and shared fitting folds limit interpretation.
[Full experiment, gates and failure analysis](source_pretrained_temporal_v1/conclusions.md).

### Annotation Episodes and Repeated Training Exposure

A fresh audit links the current 15,430 stationary-history source queries to
1,457 past-defined constant-center episodes and 545 recording-scoped tracks.
There are 207 half-box-excursion windows, but only 55 episode groups and 47 tracks;
the gates site contributes just two of these track IDs. The persistent-excursion
subset has 113 windows, 38 groups and 33 tracks. These overlapping descriptive
subsets are not additive and their groups are not statistically independent
physical events. The four physical sites remain the higher-level limitation.

Missing selected-neighbor observations do not explain this subset: all 207
half-box windows have a current and a full-history neighbor, and 206 have a
moving neighbor. Their crop coverage averages 99.879%, although annotation boxes
occupy a median 7.10 by 11.27 output pixels. Availability is not predictive utility.
Of all queries, 451 are static only on the eight sampled steps, not throughout
the intervening raw annotation frames. A further 142 sampled-static futures
contain an intervening raw change. Neither observation changes the approved grid.

Reweighting the five frozen predictors by episode or scoped track leaves all
aggregate gains negative. For centered appearance, the original -0.7622% becomes
-0.2480% under episode weighting and -0.4610% under track weighting. These are
post-hoc sensitivity descriptions, not new primary metrics or model-selection
results. [Audit and definitions](source_event_support_v1/conclusions.md).

The registered training intervention samples episodes equally within each
training complement, then rows equally within an episode. It retains every row,
the original evaluation, normalizers and per-example ADE loss. All 24 fresh heads
complete 10,000 updates. Geometry and centered-appearance equal-site gains are
-37.3268% and -54.9917%, with conditional four-site intervals
[-52.3613,-26.0828] and [-79.7777,-39.1095]. Their uniform controls, reused after
hash verification, give -0.0704% and -0.7622%. All new held fits are negative;
centered-minus-geometry is -17.6649 percentage points. Static-target harms rise
to 0.576017 and 0.765669 annotation pixels, while nonzero-target errors also rise.

Equal episode probability changes the expected loss despite retaining its
per-example formula. A post-hoc diagnosis measures future-change frequency at
38.62-47.49% under uniform rows and 61.71-73.54% under the new sampler. Future
labels are used only for this diagnosis, not group construction or model inputs.
All new heads improve their reweighted training risk over CV: 4.52-13.29% for
geometry and 9.23-21.71% for centered appearance. Yet their original unweighted
training risks worsen by 10.61-14.80% and 14.46-19.73%. This identifies a measured
objective mismatch, not a unique explanation of held-site failure or proof of
sufficient past information. More sampled exposure does not add independent
events. An importance-corrected follow-up has not run.

All 24 predictions replay exactly, weighted sampling streams regenerate, six
OOF archives recompute and completed resume preserves 84 artifacts with no new
updates. Engineering reproducibility does not establish neural benefit. The
same four explored sites/shared training complements remain a limitation; no
main, outer or independent confirmation score is introduced.
[Experiment, diagnosis and gates](source_episode_sampler_v1/conclusions.md).

### Restoring The Original Risk Under Balanced Exposure

We retain the preceding episode proposal probabilities p_i, but multiply each
sampled row's ADE by 1/(N*p_i), where N counts that fold's training rows. No
self-normalization or factor clipping is applied. The expected loss and
unclipped gradient equal those of uniform-row training; this identity does not
extend automatically to clipped gradients or Adam updates. Four real training
complements and exhaustive small-batch tests verify the identity, without
using held outcomes to construct weights.

Another 24 fresh heads complete 240,000 updates, with exact draw-stream matching to
the uncorrected control. Original uniform and uncorrected heads are hash-verified
cached controls. Geometry and centered-appearance gains become -0.03206% and
-0.27459%, with conditional four-site intervals [-0.05775,-0.01284] and
[-0.68041,-0.03179]. Relative to uncorrected episode training, this reduces harm
by 37.2947 and 54.7171 percentage points, strongly supporting objective mismatch as
a cause of the earlier large degradation. Neither corrected arm beats CV, and
all 24 held fits remain negative. The improvement is a training repair, not proof
of useful neural dynamics.

Against uniform training, corrected geometry's 0.03832pp difference has interval
[-0.00078,+0.10872]; centered appearance's 0.48757pp has interval [+0.08462,+1.06884].
Centered appearance still loses to corrected geometry by 0.24254pp. Absolute
static harms are 0.000639/0.007827 annotation pixels; percentage degradation against
zero-error CV remains undefined. Nonzero-target gains remain negative, and
binary candidate/CV oracle gains are only 0.002304%/0.064806%. No learned gate or
deployment is claimed.

All logged gradients exceed the unchanged norm cap 5; logging is at update 1 and
every 100 updates, not every gradient. This motivates a separate training-only
conditioning diagnosis, subsequently completed below, rather than attributing every
failure to clipping. All 24 forecasts replay exactly and zero-update resume preserves
84 artifacts. Same explored sites/shared folds, offline annotation and nonmetric
raw-frame limitations apply. No new main/outer or independent result is read.
[Complete comparison and limitations](source_importance_sampling_v1/conclusions.md).

### Gradient Diagnosis and Readout Conditioning

At all24frozen final checkpoints, we compute full uniform-training gradients and
6,144 minibatch-gradient probes, without new held forecasts or optimizer steps.
Clipped-mean/population cosines are at least0.99858/0.99930 for uniform/corrected
episode proposals. This does not support a large direction reversal at these
iterates. At least99.99839% of squared gradient energy is in the final layer;
training restoration-radius/loss-scale medians are527-627. Nonsmooth static ADE
can have a large gradient near zero; these values alone do not prove a bug.

A registered single-factor comparison divides the pre-bound readout by the
training-derived median scale. Multiplying the last affine weights and bias by
that constant restores the original forecast, preserving the function class and
bounds. The parameterization changes optimizer geometry, not only clipping.
All24heads complete240,000updates with matched samples, seeds and objectives.
Logged clipping falls from100% to0%, and the static forecast jitter falls sharply.
However, equal-site gains remain-0.000251%/-0.001342%, conditional intervals
[-0.000631,-0.000027]/[-0.003655,-0.000050]. Every held fit remains negative.
Nonzero-target gains are negative and candidate/CV oracle gains are only
0.0000303%/0.0003394%. Improvement against damaged neural controls is not a
forecasting advantage against the baseline, and micro-pixel differences should
not be interpreted as meaningful physical precision. No new deployment follows.

Exact replay, guarded roles, future-label poison checks and zero-update resume
pass. The four explored sites/shared folds cannot provide independent
confirmation. These findings motivate investigating genuinely informative past
motion observations, not further threshold tuning of near-zero candidates.
[Full comparison](source_conditioned_readout_v1/conclusions.md).

### Observed Box-Motion Representation And Probability Probes

A fixed comparison replaces frozen appearance tokens with observed regional
flow summaries while keeping the same source cohort and training objective.
We restore crop translation and annotation/video axis scaling, require pixel
coverage and forward/backward consistency, and distinguish annotation-box and
surrounding-region proxies. All 15,430 queries remain; 23,890 unique observed
pairs supply 108,010 overlapping pair uses. These are neither independent
events nor verified body segmentation or camera compensation.

The two matched arms use quality-only or quality-plus-motion tokens with the
same 63,960-parameter temporal readout. Twenty-four fresh fits complete 240,000
updates over four explored sites and three seeds. Equal-site ADE gains versus
stationary CV are -0.000535% and -0.000840%; conditional intervals are
[-0.001030%, -0.000164%] and [-0.001184%, -0.000302%]. All held fits are negative.
Eighteen fits improve training ADE slightly, but none transfer positively.
Motion-minus-quality is -0.000306 percentage points. These tiny differences
describe forecast jitter, not physically meaningful precision. All predictions
and all observed-flow pairs replay exactly; no model is deployed.

Sixteen separately registered logistic probes test whether the features contain
information about future annotation changes, without replacing the trajectory
endpoint. For any nonzero change, adding motion does not consistently improve
classification. For maximum excursion above 10 annotation pixels, mean AUROC
increases by 0.01241 (conditional interval [0.00667, 0.02101]), but absolute
AUROC remains only 0.407-0.556. Brier and log loss worsen on every excluded site;
both feature arms lose to a train-prevalence constant predictor on Brier.
This weak ranking difference cannot justify a reliable safety gate or demonstrate
trajectory direction. The prespecified probability-error contrast is negative.

The probe's stored float32 target reconstruction produces 739 positives, whereas
exact raw annotations produce 728. All 11 disagreements occur at exactly 10 raw
annotation pixels. The original training/results are retained, and a post-hoc
evaluation of the same frozen probabilities against raw labels is disclosed
without refitting or selecting a result. The negative Brier finding persists.

The 32px crop's median annotation box is 9.34 by 11.86 pixels, smaller than the
fixed 15px flow aggregation window. This is a measurement limitation, not proof
of the failure mechanism or a promise that higher resolution would work. The
earlier native-resolution ETH/Hotel/Zara experiment was also negative. The native
SDD follow-up below is now complete and also lacks proper-score benefit. Four explored source sites,
shared training folds and retrospective supplied annotations still do not
establish independent confirmation or strict sensor-as-of forecasting.
[Trajectory evidence](source_box_motion_v1/conclusions.md) and
[probability evidence](source_box_motion_probe_v1/conclusions.md).

### Broader Source Population And Metric Conditioning

An auxiliary-only audit retains all 175,756 past-indexed queries from the four
already-explored source sites, rather than just the stationary subset. It finds
143,918 complete, 29,039 partial and 2,799 absent future-label windows. No main,
bookstore, original validation/test or external readout is opened. Retrospective
event categories are assigned only to complete labels. These are annotation
categories, not independently verified behavioral events.

Among complete windows, 6,864 static-start queries (4.77%) contribute 99.7481%
of equal-site past-normalized CV error but 0.6660% of annotation-pixel CV error.
The past-normalization scale is floored at .001 pixels for static histories.
The difference recurs at every source site. This is not future leakage; it is
a consequential weighting of the prediction task. The seven fixed kinematic
baselines coincide on these static histories. Their complete-moving diagnostic
oracle headroom is 15.2165%, versus only .03835% on the complete aggregate.
Even perfect moving-window prediction with static predictions unchanged could
improve the latter by at most about .252%. This bounds that restricted repair,
not all possible neural predictors.

The audit independently rebuilds all raw index keys, reduces 1,230,292 baseline
cost pairs and exactly replays 256 raw geometry/label queries. Future-array
poisoning leaves observed features unchanged in 33 checks. No fitting occurs.
The old primary metric and negative results remain in force; a proposed new
native-unit/equal-scene evaluation registration awaits explicit decision. A
different score definition cannot establish a model improvement or restore
independence to exposed scenes. Pipeline correctness, forecasting accuracy and
the relevance of the chosen error weighting are separate questions.
[Complete source diagnostic](source_population_v1/conclusions.md).

### Native Resolution And Motion-Window Controls

We recover 25,300 native observed crops and verify that every supported reduction
exactly reproduces the previous RGB and coverage arrays. We compare two image
resolutions (32 and 96 pixels) and two nominal flow averaging-window extents
(15 and 45 video pixels), holding the query population and other prescribed
parameters fixed. Sampling lattice and polynomial support still differ with
resolution, so this is not a perfect isolation of spatial detail alone.

Sixty-four fixed logistic probes use geometry and quality, with or without the
regional motion vectors/magnitudes. Quality already includes consistency and
support proxies. Exact raw annotation coordinates define supervision-only labels,
avoiding the earlier float32 threshold ambiguity. For larger excursions, adding
motion worsens equal-site Brier by 0.001492, 0.001269, 0.001478 and 0.001331 for
lowpass45, lowpass15, native45 and native15. All four conditional intervals exclude
a favorable Brier difference. All motion and quality variants lose to their
training-prevalence reference on that label at all four held sites.

Training mean AUROC ranges from 0.7944 to 0.7994; held-site means range from
0.4826 to 0.4946. The small positive motion-versus-quality ranking contrasts do
not overcome poor probability error or establish a useful trajectory direction.
Native coverage/consistency support increases, but is not ground-truth body
motion. The 728 positive windows cover only 115 scoped tracks in 23 recordings;
overlap does not create independent events. Four explored sites and shared
training folds limit the conditional 2,000-resample intervals.

All 95,560 flow measurements and 64 coefficient predictions replay exactly;
completed resume preserves 486 artifacts with zero new work. This measurement
repair does not justify another full trajectory budget on the same representation.
It also does not prove that all image representations lack useful information.
Broader source support, transferable directional candidates, independent
calibration and confirmation remain necessary for the proposed intervention
method. No new neural forecasting model or deployment claim is made.
[Full native-motion evidence](source_motion_resolution_v1/conclusions.md).

### Past-Only Temporal Centering Control

A subsequent input audit aligns all 123,440 historical query keys and finds
non-identical, supported eight-frame sequences. Within-window variation accounts
for only 2.5715% of frozen feature energy on average. We test two fixed transforms
on the same 15,430 source queries: per-window mean centering, and centering with
RMS normalization. The shared component is not proven to be background, and
observed pixel changes are not verified intention cues.

All 24 new heads complete 10,000 updates; the previous 36 heads are cached matched
controls. Equal-site ADE gains versus stationary CV are -0.7622% and -1.7564%,
with conditional four-site intervals [-1.7428,-0.1231] and [-3.4247,-0.5537].
All 24 held fits are negative. Centering improves the failing sequence control
by 5.3401 percentage points but remains 0.6918 points below geometry-only.
RMS normalization further worsens centered prediction by 0.9942 points. Both
new arms also lose on nonzero targets. All registered contrasts are retained.

The repair reduces harm but does not establish useful visual dynamics. Each
fixed candidate's future-informed binary oracle is small (0.1630% and 0.3368%),
and is not a learned deployment rule. Absolute zero-target harms are retained;
percentage degradation against the zero-error floor is undefined. Twenty-four
exact replays, matched sampling and immutable completed resume establish local
reproducibility, not independent validation. No primary/main or outer role was
scored and no model was deployed. [Complete comparison and limitations](source_temporal_centered_v1/conclusions.md).

### Native Detail and Spatial Pooling Control

A registered input repair retains the original96x96 observed crops and compares
them with their exact32x32 block averages upsampled to96x96. Both use an identical
fixed flow estimator; grid-based inputs separate locality from native detail.
Shared coverage/consistency controls, network size, three seeds, three held fit
scenes and4,000updates are fixed. All36models complete on all11,966fit queries.

Quality control, lowpass central pooling, lowpass grid and native grid yield
-1.0260%,-1.0365%,-1.0810%and-1.1385%primary gains versus CV. None preserves easy
cases. Native detail versus lowpass grid is-0.05693%, with exploratory paired
scene interval[-0.99781%,0.07018%]. Exact checkpoint replay supports reproducibility,
not a positive visual contribution. Native-coordinate per-recording diagnostics
also fail. There is no new deployment or independent confirmation result.

Stationary-history native motion features exceed the fixed training-support
range in100%of ETH and90.14%of Hotel rows. These samples still represent only31
source IDs; increasing input resolution adds no independent state-change examples.
This falsifies the fixed resolution/pooling repair, not all visual forecasting.
The next source-support decision must preserve the registered evaluation roles
and keep prior SDD exposure explicit. [Complete evidence](spatial_motion/conclusions.md).

### Explicit Past Image-Motion Control

A further registered fit-only comparison tests whether explicit observed motion
supplies missing directional information. Fixed Farneback features restore crop
scale and recentering before the supplied dataset-local coordinate transform.
Matched quality-only, magnitude and directional variants use the same full cohort,
three seeds/folds and two fixed objectives. All54fits complete4,000updates each.

Under row/log training, gains versus CV are-0.996%,-0.843%and-0.972%. Under
scene/ADE+harm, they are-226.739%,-201.360%and-188.996%. None passes easy preservation.
Directional input reduces the ADE-trained quality control's error11.55%, but
the exploratory three-scene interval[-21.23%,15.64%]crosses zero and all candidates
remain below CV. This is damage reduction, not a positive forecasting result.

Stationary-history motion features exceed the fixed training-standardization
range in88.89%of ETH and71.83%of Hotel held stationary rows. Hotel stationary
histories account for99.75--99.95%of ADE-arm positive harm. High image-flow
consistency does not establish departure intent or direction. All54predictions
replay exactly; no sealed role, target definition or main metric changed.
The result argues against this compact motion-summary repair, not against all
visual forecasting. Independent state-change support and candidate quality
remain unresolved.[Full comparison](observed_motion_v2/conclusions.md).

### Matched Training-Objective Study

A registered fit-only follow-up holds the geometry predictor and input cohort
fixed while separating row/scene sampling and log1p/mean-ADE losses. A fifth arm
adds a fixed unit-weight positive-error-increase penalty relative to CV.
All45models complete4,000updates each across three seeds and three physical-site
folds. Constant training dimensions are removed from extrapolation in every arm.
The primary metric and sealed roles are unchanged.

Equal-scene/seed gains versus CV are-1.03% (row/log),-165.67% (row/ADE),-1.47%
(scene/log),-234.53% (scene/ADE), and-237.24% (scene/ADE+harm). None passes easy
preservation. In-sample primary gains reach46.40%, but transfer fails. On Hotel,
stationary-history rows account for about99.85--99.95%of the ADE-based arms'
positive error increase. All per-recording native-coordinate seed-mean contrasts
also remain negative; reporting a different unit would not establish success.

All45checkpoints replay exactly. Paired2,000scene-resample intervals are
exploratory with only three exposed fit sites, not confirmation or formal risk
coverage. The result falsifies the tested loss-only repair, not every risk-aware
method. It motivates stronger observed-state/directional support before another
selector. The oracle remains a label-only diagnostic, not a deployable result.
[Complete controlled comparison and limitations](objective_alignment/conclusions.md).

### Registered Offline Visual Information Study

The current visual comparison retains all 11,966 approved fit windows, rather
than restricting learning to the previous stationary subset. It uses three
physical-scene folds (ETH, Hotel, grouped Zara), three seeds and four matched
arms: geometry, spatial coverage masks, one current RGB crop, and eight past
RGB crops. Missing Zara03 imagery remains explicit. The targets and primary
past-normalized ADE are unchanged. Each final-update model is evaluated on a
held fit scene after 2,000 fixed updates; no separate development, calibration
or confirmation data are opened. All 36 fits completed (72,000 updates).
Equal-scene/seed gains against training-selected CV are -0.5767%, -0.5692%,
-0.6627% and -0.7401%, respectively. Every held-scene easy subset degrades.
All final checkpoint predictions replay exactly; no model is promoted.

Current/past RGB versus geometry yields -0.0855%/-0.1625%, with exploratory
2,000 scene-bootstrap intervals [-0.2608%, +1.0561%]/[-0.2657%, +0.4194%].
Only three historically used physical sites and shared training folds limit
these intervals; they are not independent confirmation. Full training loss
improves, but held-scene forecasting fails. A fixed-model constant-feature
diagnostic reduces some damage without producing gain. The 31 stationary-history
source IDs still account for 89.37% of the equal-scene CV error, while a perfect
binary CV/neural chooser has only 0.19--0.31% overall headroom. This is evidence
to improve the predictor before further gating, not a claim of impossibility
or support for the proposed joint-intervention method.
See [complete results and caveats](offline_visual_forecast/conclusions.md).

This study adopts offline annotated observations, not strict sensor-as-of
causality. Retrospective interpolation and unresolved physical video timing
are disclosed. Its purpose is to test information value, not claim that a CNN
or a baseline-plus-network architecture is novel. Existing work already examines
single-trajectory baselines, static errors and hybrid state-change recognition;
the [positioning note](offline_visual_forecast/literature_positioning.md) separates
that prior art from the unestablished joint-intervention contribution.

### 4.1 Completed Development Evidence

The frozen v1 protocol fits ETH, Hotel and Zara01/02/03, with all Zara recordings
kept in one producer fold. Development uses Students01/03, one University site.
These historically exposed data are explicitly not independent confirmation.
The task uses native annotation steps, not verified metric/time calibration.

| Seed | Neural normalized ADE gain vs CV (%) | Candidate/floor oracle gain (%) | Selected policy |
| --- | ---: | ---: | --- |
| 17 | -7.093 | 0.722 | CV floor |
| 29 | -7.900 | 0.704 | CV floor |
| 43 | -7.249 | 0.441 | CV floor |

The oracle uses future labels for diagnosis only. There are 14,920 complete
development paths, 14,931 valid endpoints and 30,438 past-supported queries;
these are distinct denominators. Fifteen neural fits complete 1,000 updates each.
One development site cannot provide a meaningful site-bootstrap interval;
seed variability is not new-site uncertainty. Easy baseline error is near zero,
so absolute excess accompanies relative degradation. No successful intervention,
coupling gain or safety guarantee follows from selecting the unchanged floor.

Fit-only audits show approximately 1% of ETH and Zara02 windows account for
96.32% and 99.74% of squared normalized target energy. Development normalized
error is also highly concentrated: the top approximately 1% of Students03 rows
accounts for 62.49% of CV normalized error but 1.09% of native-coordinate error.
This motivates a separately versioned one-factor Smooth-L1 loss ablation, not
post hoc replacement of the primary metric. Details and all failed MSE controls
are in [the v1 result package](8to12_development_v1/conclusions.md).

That ablation is now complete for the same three seeds. Primary gains versus CV
are -0.231%, -0.329%, and -0.181%, with every development choice still the floor.
Mean gain changes from -7.414% to -0.247%; seed SD is 0.428 versus 0.075 percentage
points. This is a repeatable objective repair on development data, not independent
statistical evidence for the method. The two-candidate oracle upper bound falls
to 0.231--0.351%, so further gating of these frozen candidates cannot produce
a 5% primary improvement. In native coordinates, Students03 gains versus CV do
not translate into consistent gains against a stronger damped-velocity baseline.
The [paired table](8to12_robust_v2/robust_loss_comparison.md) retains that comparison
and all negative seeds. No metric or easy threshold was changed after evaluation.

### 4.2 Continuous-Context Repair and Matched Public-Core Study

The original Students01 package uses fixed twenty-point trajectory chunks.
Every one of its 17,820 rows maps uniquely to the continuous source at the same
frame and rounded position, but 415 source identities become 891 chunk identities
and 3,993 short/tail points are discarded. In particular, all 63 source tracks
shorter than twenty points disappear. Exact retention is the largest multiple
of twenty not exceeding each source track length. This is not timestamp drift
or a future-coordinate feature; it conditions the available observation
population on later track availability. That distinction matters for full-scene
intervention, even when a packaged-window benchmark has different intended scope.

The v5 development source restores continuous identities and keeps short and
tail histories when they are past-supported. Complete Students01 8-to-12 windows
increase from 891 to 14,295, without adding a physical scene. Fit recordings,
targets, selection rules and physical folds are unchanged. A real 10,000-update
full EqMotion fit is exactly reproduced after the source-only amendment,
including every recorded training loss and model parameter. This checks
unchanged fitting on this machine, not independent prediction quality.

The registered v5 study compares fixed-head EqMotion and a local Transformer,
both using complete aligned past neighbors, Smooth-L1 loss, batch32, learning
rate 0.0003, and 10,000 updates per full/held-fold predictor. Seeds are 17/29/43;
each has a full model, three physical-fold producers, an OOF ridge cost head,
and a 1,000-update neural cost head. EqMotion runs on MPS and the local model on
CPU. Update/sample budgets are matched, not parameter counts, FLOPs or devices.
Unused EqMotion output heads are skipped only after bitwise output and
trainable-gradient equivalence checks on CPU/MPS. The selected head and parameter
initialization are unchanged. This is a K=1 adaptation, not the author's
published minADE20/minFDE20 training/evaluation protocol.

The v5 comparison is **incomplete**, not a three-seed accuracy result. Seed17
completed all fitting and development evaluation after one CPU recovery; seed29
then failed with nonfinite loss on both MPS and CPU. Its paired analysis
requires complete budgets and identical row support across models. Since v5
changes context support as well as the earlier training budget, v2-to-v5 is not
a single-factor architecture ablation. Upstream annotation construction,
independent confirmation, physical scale and effective time remain unverified.

A separate fit-only input audit finds ego-history norms at most one but aligned
neighbor norms up to 16,447. In Zara02, 2,033 of 5,741 supervised fit windows
have a neighbor norm above 100. This is an input-conditioning hypothesis,
not measured per-example gradients or proof of a neighbor-induced error.
No development labels were accessed and the current transform, metric and
training budget were not changed. Any input-conditioning repair must be a
separate ablation that preserves the evaluated error scale.

The v6 numerical repair divides the two predictors' coordinate inputs by the
largest observed ego/aligned-neighbor radial norm, lower-bounded by one, and
multiplies their outputs back before the original loss and evaluation. No target,
future-valid mask, sample membership, parameter count, policy or error scale is
changed. The formerly failing seed29 completed a real MPS 100-step pilot in
13.01 seconds before the complete-budget experiment below. Finite early
training is not evidence of downstream lift. v5 remains visible as a failed
comparison rather than being merged with v6 results.

Both v6 families now complete all three seeds, with 24 forecasting fits at
10,000 updates, six neural cost heads at 1,000 updates and six OOF ridge controls.

| Seed | Transformer normalized-ADE gain vs CV (%) | EqMotion-K1 gain vs CV (%) | Both selected |
| --- | ---: | ---: | --- |
| 17 | -5.736 | -14.119 | CV |
| 29 | -7.686 | -8.473 | CV |
| 43 | -6.700 | -14.081 | CV |

Every diagnostic candidate/CV oracle gain is below 0.61%. Native-coordinate
EqMotion gains over the development-best causal alternative on Students03 are
2.103%, 2.318%, and 0.940%, but Students01 remains negative. This is not a change
of primary metric or new-site evidence. Numerical stability improves at the
registered budget, while predictor and intervention claims remain unsupported.
The paired table preserves 28,324 complete ADE paths, 28,335 valid endpoints and
37,775 past-supported agent queries as distinct denominators.

A post-run decomposition keeps the fixed metric and every row. The 3,082
complete queries at its numerical scale floor constitute 10.88% of the scored
population but contribute 88.88--91.58% of Transformer positive harm and
93.08--96.97% of EqMotion positive harm. These are descriptive error shares,
not causal attribution. Net excess error is also positive above the floor;
removing low-motion rows cannot be presented as a successful method. This
motivates a prospective motion-state/scale hypothesis, not retrospective
replacement of evaluation or selection rules.

### 4.3 Baseline-Relative Output Ablation

The prospective v7 comparison retains v6 data, primary metric, loss, folds,
seeds, update budget and policy thresholds. Both new arms initialize the last
output layer to zero and add the network output to the declared causal CV path.
The bounded arm alone maps each restored residual d to
A*u*d/sqrt(1+||d||^2), with A=max(observed ego path length, maximum requested
CV displacement) and u=requested time / maximum requested time. These quantities
use only the observed past and requested forecast. Neighbor distances and target
easy labels do not set the radius. Numerical rescaling evaluates the same map
without overflow; no new trainable parameter is introduced.

Both arms complete three seeds, four forecasters per seed at 10,000 updates,
and the unchanged OOF ridge/neural cost heads. Mean primary gain is -0.5986%
for the unbounded CV-skip and +0.06151% for the bounded arm. The latter has
positive gain in every seed but easy degradation of 62.58%, 398.57% and 308.07%
without selection. Under the unchanged development rule, selected gains are
only +0.002495%, +0.007710% and +0.000519%, with easy degradation 1.189%, 1.883%
and 0.251%. These policies are selected and assessed on development data, differ
by seed, and do not constitute an independently confirmed policy.

The bound gives zero extra error on the existing numerical-floor slice, while
missing starts after a stationary past by construction. A fit-only per-step
ball oracle quantifies this capacity tradeoff: 365 zero-budget rows contain
73.253% of pooled fit CV error; optimistic fit-window headroom is 4.089%.
This label-aware diagnostic is not a learned or test result. Bounded predictor/CV
oracle headroom on development is at most 0.337%, so further gating of these
unchanged candidates cannot produce a 5% primary gain. The
[complete results](8to12_residual_pair_v7/conclusions.md) retain native-coordinate
causal comparisons and all negative controls. A bound on output magnitude is
not a bound on realized excess loss, a novel safety theorem or a physical claim.

The fixed v7 raw50/count-matched supplement is also complete. CV-skip has
identical matched identities throughout. The bounded arm has nine zero
comparisons and three small ADE increases favoring independent selection;
196 repeated agent-query switch records differ. Uncontrolled bounded raw50
ADE gains are +0.04217%, -0.04695% and +0.00559%. Full primary error exports and
ordinary decisions replay exactly. Thus the output repair does not establish
the proposed joint mechanism, and the supplement does not select another model.

### 4.4 Fit-Only Stationary-Start Identifiability

A source-verified diagnostic follows the zero-budget result without changing the
forecasting task or primary metric. The 365 stationary fit windows are 31 agents
and 45 runs in ETH and Hotel; none occur in the three Zara fit recordings. Exact
source/cache agreement rules out corruption of these cached positions, but does
not establish physical stillness, annotation causality or clock calibration.
Labels indicate any coordinate change in the next twelve native steps, not
verified physical intention. Changed-future prevalence is 72.84% in ETH and
45.42% in Hotel, so the two directions are materially different learning tasks.

Twenty-four fixed logistic/tree classifiers with past-only ordered neighbor
features all worsen window-level Brier against the opposite-scene training-only
prior. One adaptive single-factor repair pools neighbor context, retaining rows,
folds, labels, models and settings. Pooled geometry ExtraTrees improves Hotel->ETH
Brier by 0.02513 (mean AUC 0.6949), but worsens ETH->Hotel Brier by 0.01943 (AUC
0.4895). These are absolute Brier differences, not forecast improvements. The
positive held direction has only five agents. Adding motion summaries does not
produce positive Brier in either direction. Every setting is retained in the
[full diagnostic](stationary_start_probe/results.md).

Run/agent-balanced summaries keep the local positive signal, without establishing
independence. All 48 fitted models replay their saved scores; seed repetition
does not create more independent sites. This adaptive fit-only evidence supports
neither a new stationary-start trajectory head nor a confirmation claim. Scene
cues and start-direction prediction remain untested. More context, pooling and
non-collapse alone are not evidence for useful world dynamics.

### 4.5 Remaining Mechanism and Confirmation Tests

A further [fit-only static-scene experiment](stationary_scene_probe_v2/conclusions.md)
adds supplied obstacle geometry, but not image pixels or original destination/group
labels. We retain the 365 stationary windows and physical folds. Each version fits
36 start classifiers and 36 multi-output trajectory regressors. A versioned repair
removes a shared-corner ambiguity error affecting 35 Hotel reference frames; all
144 saved models across both versions replay their predictions. The original
version's two small positive guarded results disappear after the correction.

Corrected static-map trees improve average Brier versus the training-only prior
by 0.00617 in ETH (AUC0.8048) and 0.02894 in Hotel (AUC0.5387). However, all36
corrected regressors worsen stationary-subset ADE versus CV, and the fixed0.9
probability gate has no positive trajectory result. Start probability and useful
direction/displacement are distinct tasks. All easy rows in this subset have
zero CV error, making percentage degradation undefined; absolute excess remains
visible. All rows share the same past-scale floor, so these negative percentage
gains also hold in native coordinates. They are not explained away by units.

The maps are unverified static proxies. Reference images contain people and have
unknown capture times, and no image/annotation synchronization was established.
More reliable scene semantics, directional evidence and independent site support
remain gaps. This diagnostic does not establish a scene-aware dynamics contribution
or change the primary benchmark, policies, calibration or confirmation status.

A [label-resolution follow-up](stationary_label_resolution/conclusions.md)
checks the same fit-only windows without fitting another predictor. None of
188 changed windows is compatible with a constant value under printed decimal
intervals. Supplied-H projections reveal integer-pixel lineage, but changes
above five inferred pixels still account for 94.26% of ETH and 81.29% of Hotel
stationary-subset CV error. Unrestricted regressors remain negative on this
fixed slice; a small guarded Hotel slice gain does not survive full-subset
evaluation or protect still cases.

We also test a specific conditional explanation: a constant-velocity path in
native coordinates whose projections lie within +/-0.501 pixels of every
recorded location over eight observed and twelve future steps. Four-variable
linear feasibility programs reject that model for 46/59 changed ETH windows
and 115/129 changed Hotel windows, accounting for 98.30% and 97.21% of their CV
error. All 177 unchanged windows are feasible; no solver case is inconclusive.
This label-side oracle rejects only hidden CV plus the specified rounding model,
not arbitrary annotation error. It supplies no input feature, physical-motion
proof, causal start predictor or new deployment. These overlapping fit windows
are not independent confirmation, and no target or metric is changed.

A [source clock and matrix audit](annotation_clock_geometry/audit.md) finds that
ETH's six-frame annotation spacing conflicts with a naive combination of the
25-fps video header and the documented 0.4-second annotation interval. Students03
pixel rows reproduce the stored coordinates under H-old, not the currently named
H matrix; thirteen additional source rows remain explicitly unmatched. These
observations do not establish physical calibration. No experiment is relabeled
as seconds-level or metric, and the registered data and policy are not changed.

A [past-video admission audit](past_video_alignment_v2/conclusions.md) retrieves
62 first-past/current image requests for 31 fit agents (48 distinct frames).
The upstream plotting convention swaps inverse-H axes; applying it changes Hotel
in-frame coverage from 82.78% to 99.83%, without changing any trajectory forecast.
Fixed inspection rectangles are not verified body boxes, and occlusion, identity
registration and ETH source-clock ambiguity remain. Thus this is a corrected
media-access diagnostic, not a visual-feature ablation or new multimodal result.
No image-based model was trained and no physical-time claim follows.

A subsequent [moving-control check](past_motion_comparison/conclusions.md) uses
all eight observed images for 24 fit agents, without opening future target labels.
ETH's local correspondence is often within a few pixels. Hotel's initial errors
are partly an image-search capacity artifact: 56/84 annotated displacements lie
outside the fixed search square. Increasing the radius improves Hotel error on
identical support but worsens ETH through distractors and loses boundary support.
This supports native-index plausibility, not a calibrated pose estimator, physical
time mapping or image-based forecasting gain. Centered appearance with explicit
visibility masks was subsequently tested as described below, not validated by
the correspondence audit alone.

The [subsequent appearance forecast ablation](past_appearance_probe/conclusions.md)
fits18 small neural models: geometry/current-RGB/eight-past-RGB, three seeds and
two held fit-scene directions, each at1,000 updates. All365 stationary rows remain,
including32 incomplete image histories with explicit masks. No guarded setting
has positive ADE gain versus CV. ETH guarded seed means are -0.01/-8.09/-14.98%;
Hotel means are -132.04/-167.73/-172.63%. All18 saved predictions replay exactly.
Train-to-held gaps and asymmetric covariate support are substantial: ETH has
only five stationary training agents, and78.52%of Hotel rows exceed at least one
ETH-standardized feature clamp. Start probabilities are worse than the training
prior on both held scenes. This small diagnostic rejects the current appearance
predictor, not visual conditioning in general. Its0.9 gate is not calibrated
risk control; exact-zero still-row CV error makes percentage easy degradation
undefined, so absolute harm is reported. No independent scene CI, final-test
success, primary-metric revision or new deployment is claimed.

A [frozen input-support control](appearance_support_control/conclusions.md)
evaluates all18 predictors under four registered treatments without refitting.
Training-range projection reduces some Hotel harm but no guarded setting beats
CV; strict support abstention rejects every held query. A same-original-mask
decomposition distinguishes trajectory changes from fewer interventions. For
geometry-only Hotel, camera clipping changes guarded gain from -132.04% to
-98.12%, yet the fixed-mask result is -135.32%. Thus that apparent improvement
does not demonstrate better dynamics. Marginal training support is neither
joint-distribution support nor a statistical safety certificate.

[Six matched retrains](appearance_no_camera/conclusions.md) then remove only
the four learned camera-Jacobian inputs, preserving output coordinate mapping,
all other features, initialization, sampler, loss and1,000-update budget. Past-RGB
guarded seed-mean gain changes from -14.98/-172.63% to -19.64/-86.28% onETH/Hotel.
All six remain negative and replay exactly. This rejects a simple camera-input
repair, not visual conditioning in general. The31-agent/two-site diagnostic has
insufficient independent support for a generalization or risk-control claim;
additional threshold sweeps would not establish the proposed contribution.

A [controlled point-decision experiment](conditional_ade_probe_refined/conclusions.md)
reconstructs the conditional training distribution from each frozen ExtraTrees
forest and compares its mean with per-step geometric medians, holding all trees,
features, fit folds and gates fixed. All18 mean forecasts replay within1e-12.
All nine ETH median settings predict zero, equaling CV without improvement.
Hotel seed-mean gain changes from -346.70/-213.07/-252.23% to
-90.38/-21.37/-21.25% for pooled/static/directional-neighbor features. No median
or fixed-gated median improves the full stationary subset. Optimizing a more
appropriate point decision reduces false movement but cannot correct a poor
cross-scene conditional distribution. This is not a new forecasting architecture.

Of39,420 repeated waypoint computations,23 initially missed the strict numerical
gap tolerance; training-only refinement certifies22 of those. One remains
approximate with gap bound2.293e-9, explicitly retained. No held result selects
the solver or threshold. The all-setting signs remain unchanged. Percentage
easy degradation is undefined on zero-error still rows, so absolute harm is
reported. These fit-only comparisons do not supply independent confirmation
or a physical safety guarantee; future labels never enter query features.

The [v6 forecast supplement](forecast_supplement_v6_decision.md) is fixed before
v6 development results. It applies exact-count routing and exact raw-frame t+50
prefix scoring to every frozen seed/head/policy, without selecting a new model.
The prefix uses the same original 12-step prediction and past normalization;
it is not a separately horizon-conditioned t+50 model. Implementation checks
pass and real supplementary scoring is complete. All joint raw50 ADE gains
are negative. The prefix has 33,686 complete paths and 33,690 valid endpoints;
full-path-defined easy labels are not redefined using the prefix. The supplement
cannot repair the lack of independent confirmation sites.

Transformer's twelve matched-count comparisons have identical switch identities
and zero ADE/FDE differences, including nonzero-intervention queries. EqMotion
has ten zero comparisons and two seed29 ridge differences: joint-minus-independent
ADE +0.00000904878 for conservative and -0.0000124400 for moderate. Only 76
agent-query switch records differ across twelve repeated exports. There is no
stable joint advantage. Full primary errors and ordinary control decisions
replay identically for both families. This compares exported errors/decisions,
not every intermediate tensor; matching occurs before future-label filtering,
so scored-only coverage and realized risk need not be matched. See the
[complete supplement](8to12_public_predictors_v6/supplement_conclusions.md).

The primary mechanism test holds candidate forecasts and training examples fixed while varying cost supervision and joint selection. At matched actual intervention counts, improved forecast composition would support a narrower contribution than a new predictor architecture. A gain that disappears after matching counts, or a lower proximity penalty accompanied by worse forecasting, would not support that claim. Real accuracy, independent-scene risk calibration and physical safety remain separate questions; none is established by the analytical examples in the assumption audit.

The [deferral control](deferral_control/method_and_limits.md) fits linear or small neural routing on the same causal rollout features and held-fold predictions as the relative-cost head. It preserves continuous error weights and makes its bounded-cost transform explicit in the protocol. Clipped training risk, unclipped forecasting error and calibrated safety are distinct quantities. Its [development comparison](deferral_development/implementation_and_limits.md) now verifies identical OOF training inputs and runs all controls on identical forecasts before labels are read. The unconstrained deferral arm is not claimed to share M3W's budget or coverage. Paired scene-level error differences are descriptive; synthetic integration and recovery do not establish a real accuracy advantage. The real protocol and frozen confirmation family have not been changed.

An [EqMotion-core adapter](public_baselines/compatibility_and_limits.md) provides an externally sourced predictor for this implementation path. The source is pinned and checked before loading; past-only context replaces the release's future-availability-dependent preprocessing. The complete v6 run trains one fixed output head with Smooth-L1, as specified above. It is an adapted K=1 control, not a reproduction of published best-of-20 results. All three seeds complete the matched-context, matched-budget study reported above. AgentFormer's author-reported normalization correction must also be respected when constructing the eventual public-baseline table.

Compare the same predictor with no gate, independent confidence gating, expected-error gating, scene-uniform gating and joint intervention selection. Include simple regressors and modern published predictors so that benefits cannot be attributed only to replacing a weak baseline. Separate K=1 from best-of-K evaluation. Deduplicate underlying recordings across dataset distributions, freeze development and calibration choices, and use a final confirmation set with no prior model-selection exposure. The current development primary is past-normalized ADE; native-coordinate ADE/FDE, easy/hard slices, proximity proxies, coverage and latency are complementary, not post hoc replacement metrics. Report all three seeds and only use scene-level intervals when independent scene support exists.

The [frozen final-family evaluator](confirmation_evaluation/implementation_and_limits.md) now checks actual predictor and out-of-fold producer seeds, consumes completed calibration decisions, and reports all prespecified controls without final-set model selection. Seed-mean errors are distinguished from ensemble predictions; positive harm is computed within each seed before averaging, and easy-case preservation remains visible per seed. Paired bootstrap draws share physical-scene blocks across arms and seeds, conditional on the fixed models. They are descriptive, not multiplicity-adjusted guarantees. Synthetic three-seed training and final-evaluation recovery have been exercised. Real independent confirmation remains unrun because its protocol and eligible independent support are unresolved; this does not negate the approved development and fit-only experiments reported above.

No confirmatory results are available for the proposed method. The historical Stage44 no-scene result (+37.49% all, +20.32% t50 normalized four-waypoint ADE vs an interpolated floor) is motivation only and must not appear as a final main-table result without a corrected independent evaluation.

## 5. Limitations

The [Zara source trace and repaired media adapter](zara_past_media/conclusions.md)
also distinguish annotation-timestamp forecasting from strict sensor-as-of
forecasting. Both recordings numerically reconstruct from sparse VSP controls;
Zara02 needs a fixed origin translation derived from one exact source anchor.
This does not establish physical calibration. Of3,988/8,110 complete eight-step
pasts,3,877/7,924 depend on an interpolating control later than the query.
An input API that excludes explicit future labels therefore does not establish
online observation availability. Existing results retain their offline supplied
annotation meaning; no retrospective row filter or new performance claim is
introduced. A stricter source-as-of task would need a separately approved
observation protocol. The two videos represent one physical scene.

A [masked past-image reader](zara_masked_images/conclusions.md) now retains
partially visible boundary crops without recentering or filling missing pixels.
All12,098 complete past histories remain available as diagnostic inputs, including
1,935 with partial support. Exact input-cache replay is engineering evidence;
this repair does not resolve retrospective annotation provenance or establish
new visual predictive benefit.

The current datasets are represented in pixel or dataset-local coordinates with unverified cross-source scale and effective time. Physical collision risk cannot be inferred from a normalized proximity threshold alone. Scene proxies are not verified semantic maps. Most historical observations overlap in time, and some named dataset collections may contain the same underlying recordings. Calibration assumptions may fail under arbitrary shift, and a small number of independent scenes may make safety bounds uninformative. A selective predictor is not an action-conditioned simulator or a foundation world model.

The new [support audit](risk_calibration/support_audit.md) makes this limitation concrete: nine current canonical recordings correspond to six physical-scene groups, all historically development-exposed. With six hypothetical independent calibration scenes and zero observed [0,1] loss, even one policy/one risk gives an upper bound of 0.4996 at illustrative delta=0.05 under the current Hoeffding screen. This is a sensitivity calculation, not an actual calibration or universal sample-complexity lower bound. Repeated windows cannot improve independent-scene support. Until data roles and independent confirmation are resolved, useful formal risk control must remain an unestablished part of the proposed contribution rather than a result.

A separate [CITR diagnostic conversion](citr_causal_intake/implementation_and_limits.md) now retains synchronized pedestrian and vehicle raw positions with row-level provenance. Its 38 controlled clips cover only one physical site, so the conversion cannot be counted as 38 independent calibration scenes. No data-use role, independent-test eligibility, typed neural training or predictive result has been established for this cache. It may support future controlled mechanism checks after review, not a current external-generalization claim.

The subsequent [DUT diagnostic intake](dut_causal_intake/implementation_and_limits.md) adds 28 natural-campus clips at two source-described locations, not 28 independent scenes. Exact source-row checks identified two simultaneous pedestrian IDs with identical 145-frame trajectories in one clip; that recording is flagged for quality quarantine before formal use. Raw-coordinate semantics also require care because the author preprocessing divides raw positions by a scale despite a general meter statement in the README. Source terms, scientific roles, annotation resolution and prior-use eligibility remain open. This acquisition supplies neither a confirmatory performance result nor enough independent scenes to establish the proposed risk guarantee.

A separate [admission check](intake_admission/implementation_and_limits.md) now prevents a scientific-role declaration from clearing pending source review or a bound annotation quarantine. It follows the source, conversion and quality evidence and checks role-specific review declarations before opening new-source data through the experiment contract. These engineering controls do not authenticate permission or establish independent sampling, and they are not offered as a methodological contribution or evidence of forecasting improvement.

## 6. Reproducibility

This revision reuses the stored reports; it performs no new fitting, image
extraction, inference replay or bootstrap. The motion-information report is
hash-verified as `c1d377b37aa85931078e8439ddf6ccfcc0875a52a0beb0d27ff80dbb74d335b6`;
the SDD multimodal report as
`0baf077c4c1b67ffa7d645fa8cdb4757c44d86a797457473625e310216c7f7d8`.
Their linked experiment records distinguish original fresh runs, cached
verification and work not run. No sealed labels, training role, thresholds,
primary metric or deployment policy are changed by this manuscript revision.

The completed v1 package stores canonical recording IDs, hash-bound protocol,
past-only schemas, train-only preprocessing, full held-fold producer lineage,
fixed development policies, seeds, atomic checkpoints and heartbeat logs.
Commit `707d4017` preserves its exact training implementation. Fresh real fitting
is established, but independent calibration and confirmation are not. Source
identity is checked on resume; old model hashes must not be edited to bypass a
newer implementation mismatch. Subsequent ablations use new protocol versions.

A later [completed-resume repair](cost_completion_v2/repair_and_verification.md)
addresses a narrower integrity gap: the old ridge completion branch did not
revalidate its OOF caches and full report provenance. The defect is reproduced
on temporary synthetic training. A versioned entrypoint preserves the frozen
trainer while binding and rechecking completion dependencies; a separate
read-only verifier checks all six real v6 ridge heads against earlier snapshots.
No real corruption is found, and no forecasting score changes. The feature
identity alone cannot anchor targets because it deliberately excludes them.
Missing completion receipts are not silently regenerated from current files.
These are reproducibility controls, not predictive evidence or a new method.

## Verified References

- AgentFormer: https://ye-yuan.com/agentformer/
- EqMotion: https://openaccess.thecvf.com/content/CVPR2023/html/Xu_EqMotion_Equivariant_Multi-Agent_Motion_Prediction_With_Invariant_Interaction_Reasoning_CVPR_2023_paper.html
- SingularTrajectory: https://arxiv.org/abs/2403.18452
- Joint Metrics Matter: https://arxiv.org/abs/2305.06292
- Conformal Risk Control: https://arxiv.org/abs/2208.02814
- Learn then Test: https://arxiv.org/abs/2110.01052
- SODA-MPC: https://proceedings.mlr.press/v283/contreras25a.html
- Generalized HCP: https://arxiv.org/abs/2608.15500
- HCP, inspected preprint v4: https://arxiv.org/abs/2306.06342v4
- CAFHT, ICML 2024: https://proceedings.mlr.press/v235/zhou24l.html
- Adaptive CP for motion planning, L4DC 2023: https://proceedings.mlr.press/v211/dixit23a.html
- Consistent Estimators for Learning to Defer to an Expert: https://proceedings.mlr.press/v119/mozannar20b.html
- Regression with Multi-Expert Deferral: https://proceedings.mlr.press/v235/mao24d.html
- Selective Regression under Fairness Criteria: https://proceedings.mlr.press/v162/shah22a.html
