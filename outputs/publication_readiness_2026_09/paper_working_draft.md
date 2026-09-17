# When to Trust Neural Motion Forecasts: Baseline-Relative Joint Intervention for Multi-Agent Forecasting

Working draft, 2026-09-17. Method proposal with completed three-seed development
experiments, including a matched Transformer/EqMotion study and a completed
baseline-relative output ablation. The latter reduces drift but yields only
tiny guarded development gains. None supports a new deployment or the proposed
joint-intervention contribution claim.
The completed v6 input-conditioning protocol retains the repaired v5 source population.
The older results below retain their original, conditional observation population.

## Abstract

Multi-agent motion predictors are commonly evaluated by average trajectory error,
yet an improved average can conceal degradation on motion already well predicted
by a simple baseline. We investigate baseline-relative selective intervention:
learning benefit and harm from out-of-fold predictions and selecting neural
replacements over an observed interaction graph. An initial eight-observed,
twelve-predicted-step development study uses three training seeds and physically
grouped crossfit folds. All seeds select the causal constant-velocity floor;
uncontrolled MSE-trained forecasts worsen the prespecified normalized ADE by
7.09--7.90%. A single-factor Smooth-L1 retraining reduces this degradation to
0.18--0.33% but still selects the floor in every seed. Joint and independent
controls do not differ at the evaluated policies. We identify strong sensitivity to small past normalization scales,
without treating a favorable alternative metric as confirmation. The method's
claimed advantage remains unestablished. A subsequent matched-context,
10,000-update-per-fit comparison completes three seeds for a Transformer and
fixed-head K=1 EqMotion. Their uncontrolled normalized-ADE changes versus CV
are respectively -5.74 to -7.69% and -8.47 to -14.12%; all six selections retain
CV. Numerical input conditioning resolves the observed fit failures, not the
prediction or easy-preservation problem. Completed actual-count-matched controls
find no stable joint advantage, and all joint raw50 ADE gains are negative.
A paired output ablation then changes mean uncontrolled primary gain from
-0.599% for a CV-initialized residual to +0.062% with an observed-motion bound.
Easy degradation still fails without selection; guarded gains are below 0.008%.
A real fixed-predictor deferral comparison completes 24 routing heads on the same
three seeds and OOF inputs. Bounded deferrers have small average gains
(+0.020--0.084%), but easy degradation is 14.40--96.96%; none meets the 2% ceiling.
Their lack of the guarded controls' budgets prevents attributing the difference
solely to the objective. Broader independent scenes and a positive contribution
remain necessary before a submission claim.

An upstream source audit additionally identifies identity fragmentation and
future-availability-conditioned observation retention in one packaged development
recording. We retain the earlier results as conditional-window evidence and
rebuild full-scene context from the continuous source before the next comparison.
This repair does not make previously explored data an untouched test set.

## 1. Introduction

Strong causal predictors can be difficult to improve consistently. A neural predictor may capture turns or interactions while performing worse on straight, low-variance motion. Replacing the baseline everywhere therefore asks a different question from deciding where the neural model has useful information. The latter question is especially relevant under scene shift, where a global average provides limited guidance about the risk of a local intervention.

Multi-agent prediction introduces another difficulty. Per-agent decisions can combine incompatible futures even if each predictor produces a coherent scene when used alone. We propose to treat the intervention vector itself as a structured prediction and to evaluate its excess error relative to a fixed baseline.

The study tests whether baseline-relative loss supervision and joint intervention improve mixed forecasts beyond cost-aware routing alone, and whether that effect survives dependence-aware evaluation. Regression deferral and joint compatibility already have substantial prior work. The contribution cannot be established by renaming those components; it requires matched controls, a defensible methodological distinction and new independent results.

## 2. Related Work

AgentFormer jointly models social and temporal structure with agent-aware attention (Yuan et al., ICCV 2021). EqMotion introduces equivariant motion prediction and invariant interaction reasoning (Xu et al., CVPR 2023). SingularTrajectory studies a unified representation across trajectory-prediction tasks (Bae et al., CVPR 2024). These works preclude claiming that temporal attention, relative geometry or a common motion representation are novel by themselves.

Joint Metrics Matter studies joint forecasting errors and collisions (Weng et al., 2023). The present proposal must demonstrate a benefit beyond adding a joint metric or training penalty: it concerns which predictions from competing predictors can be selected together.

Cost-sensitive expert deferral predates this proposal ([Mozannar and Sontag, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html)). [Mao, Mohri and Zhong (ICML 2024)](https://proceedings.mlr.press/v235/mao24d.html) explicitly study regression deferral with a fixed predictor. Our cost heads therefore require comparison to established deferral objectives, not only confidence thresholds. The [real two-action comparison](8to12_deferral_v7/results.md) is now complete on both frozen v7 predictors, three seeds, linear/width64 heads and two fixed cost bounds. Every result is retained without selecting a new winner. This task-specific control is not a reproduction of Mao et al.'s reported experiments.

Selective-regression work also shows that reduced coverage need not protect every subgroup ([Shah et al., ICML 2022](https://proceedings.mlr.press/v162/shah22a.html)). Our easy-error slice is a different construct, but the warning motivates reporting slice-specific damage rather than treating reduced intervention as a guarantee. The [source-scoped review](joint_intervention/deferral_and_coverage_prior_work.md) distinguishes these established results from our remaining hypotheses.

Conformal Risk Control and Learn then Test provide established tools for controlling losses or selecting risk-constrained policies. SODA-MPC combines conformalized OOD monitoring with reachability fallback in control. We target excess forecasting loss relative to a fixed predictor; we do not claim formal physical safety from low ADE.

[HCP](https://arxiv.org/html/2306.06342v4) requires both across-group and within-group exchangeability. [GHCP](https://arxiv.org/html/2608.15500v1) uses initial target-group labels under additional sampling assumptions; it is not a general repair for serially dependent windows. [CAFHT](https://arxiv.org/html/2402.09623v2) permits dependence within trajectories but requires exchangeability across trajectories for simultaneous path coverage. [Adaptive CP for motion planning](https://proceedings.mlr.press/v211/dixit23a/dixit23a.pdf) studies online coverage and feasible control. None of these guarantees alone orders a selected point forecast against our baseline. The [assumption audit and falsifiable claim matrix](joint_intervention/statistical_assumptions_and_claims.md) separate these tasks and specify which comparisons remain missing.

## 3. Method

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

## 4. Experiments To Complete

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

The [frozen final-family evaluator](confirmation_evaluation/implementation_and_limits.md) now checks actual predictor and out-of-fold producer seeds, consumes completed calibration decisions, and reports all prespecified controls without final-set model selection. Seed-mean errors are distinguished from ensemble predictions; positive harm is computed within each seed before averaging, and easy-case preservation remains visible per seed. Paired bootstrap draws share physical-scene blocks across arms and seeds, conditional on the fixed models. They are descriptive, not multiplicity-adjusted guarantees. Synthetic three-seed training and final-evaluation recovery have been exercised; the real comparison remains unrun under the still-unapproved protocol.

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

The completed v1 package stores canonical recording IDs, hash-bound protocol,
past-only schemas, train-only preprocessing, full held-fold producer lineage,
fixed development policies, seeds, atomic checkpoints and heartbeat logs.
Commit `707d4017` preserves its exact training implementation. Fresh real fitting
is established, but independent calibration and confirmation are not. Source
identity is checked on resume; old model hashes must not be edited to bypass a
newer implementation mismatch. Subsequent ablations use new protocol versions.

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
