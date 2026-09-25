# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling.

The question behind the project is simple:

> If I can see a scene, the agents in it, their recent motion, and their local interactions, can I predict what happens next more reliably than strong causal motion baselines?

I started this repo to answer that question carefully, not just to collect a nice-looking demo. The work here includes the models that improved results, the ones that failed, the leakage checks, the safety rules, and the notes that keep me honest about what the evidence does and does not prove.

## Read the Current Study

I am now testing [what the policy bridge actually contributes](outputs/publication_readiness_2026_09/european_bridge_attribution_v1/registration.md).
The experiment compares neural and ridge scoring on identical forecasts and
matches intervention counts at each query. A separately retrained motion-only
version tests whether neural trajectory candidates are necessary. All 36 new
neural heads and 36 ridge controls have finished training. All 396 policy views
are frozen before outcome readout; no new accuracy result or deployment is claimed.

I trained a [policy bridge](outputs/publication_readiness_2026_09/european_dual_event_bridge_v1/conclusions.md)
to learn when a conservative forecast should give way to a more accurate, but
riskier, alternative. The key change is that predicted gain and harm now refer
to exactly the same pair of delivered forecasts.

On the six opened model-selection localities, the single-risk bridge improves
ADE over the previous conservative controller by **3.95%-5.57%** across all
18 source-role/seed settings. All settings preserve easy cases within the 2%
net-degradation limit; the worst is 0.73%. The experiment includes three seeds,
54 real Torch cost-head fits, matched ablations and locality-level bootstrap.
[Full results](outputs/publication_readiness_2026_09/european_dual_event_bridge_v1/results.md).

There are two important limits. Adding a second risk constraint is too
conservative and removes useful predictions. More importantly, good net error
does not certify individual harm: realized conditional harm still exceeds the
predicted cap in some scene views. Most of the improvement comes from choosing
between existing motion forecasts, so I am not calling this a new neural
dynamics result. I am keeping the [negative ablations and failure accounting](outputs/publication_readiness_2026_09/european_dual_event_bridge_v1/failure_analysis.md)
alongside the gains. Independent calibration and confirmation remain closed;
no deployment is promoted. These are pixel-coordinate, annotation-step results.

### Preceding Six-Locality Readout

I evaluated the [frozen policy family on six new model-selection localities](outputs/publication_readiness_2026_09/european_selection_readout_v1/conclusions.md),
using 28 recordings and 38,102 indexed targets. Predictions and decisions were
committed before outcome evaluation. No model or threshold was refitted.

The rule that only adds interventions improves average ADE over the old controller
in all 36 configurations, by +0.0105% to +0.7790%. But the full family is not ready
to deploy: seven configurations fail the easy-case guard, with worst-locality
degradation reaching 8.22%. The distinction between risk targets matters. All 18
easy-risk configurations preserve the observed easy means, while seven of the 18
all-risk configurations do not. The conservative branch still sometimes loses to
the training-selected motion baseline, so I am not treating this as a solved problem.

The [results](outputs/publication_readiness_2026_09/european_selection_readout_v1/results.md)
include all three seeds, locality-level bootstrap intervals and every adverse
configuration. The [failure accounting](outputs/publication_readiness_2026_09/european_selection_readout_v1/failure_analysis.md)
shows why counting successful interventions is insufficient: a few larger errors
can outweigh many small gains. That failure motivated the policy-bridge study
above, not a search for a favorable seed. Twelve calibration localities and
six confirmation localities remain closed. These six opened selection localities
can never become confirmation data. No deployment or physical-safety claim.

Earlier sections below describe the data-access state at the time of each study;
only the new selection role has since been opened.

### Completed Joint-Control Study

I completed a [joint incremental-control comparison](outputs/publication_readiness_2026_09/european_incremental_joint_v1/conclusions.md)
to test whether coordinating extra interventions improves the existing controller.
All 252 policy views use frozen predictions and scores. Independent, unary,
joint and hash-priority controls share the same addition count and predicted-risk cap.

This version does not help enough. Joint selection has no supported accuracy
advantage over independent gain ranking or unary geometry. It changes only three
unique queries relative to the unary control. Restricting additions also loses
useful predictions: worst-locality easy degradation reaches 2.255%, above the
2% gate. The [failure analysis](outputs/publication_readiness_2026_09/european_incremental_joint_v1/failure_analysis.md)
shows exactly where removing beneficial choices outweighs avoiding harmful ones.

The [full comparison](outputs/publication_readiness_2026_09/european_incremental_joint_v1/results.md)
covers 6,116 rows at 1,152 fixed queries, not the full parent population.
These are dependent development views, not independent confirmation or new
neural training. Deployment stays unchanged. I am prioritizing incremental
risk reliability and independent calibration over further tuning of this
proximity penalty. All 417 scoped tests pass, alongside independent checks of
55,296 decision-budget constraints. Reserved sources remain closed; no metric
or safety claim.

### Completed Incumbent-Relative Study

I completed an [incumbent-relative intervention study](outputs/publication_readiness_2026_09/european_incumbent_relative_v1/conclusions.md):
can a controller learn when an existing decision is worth overriding, without
discarding useful predictions? I trained 144 small Torch cost heads and 72 ridge
controls on identical forecasts and causal inputs, then evaluated all 288 frozen
policy views across three seeds and six source-role rotations.

Preserving the original decision helps. All-ADE changes against the incumbent
range from -0.0089% to +1.1520%, with 33 positive and one negative confidence
interval. A predeclared rule that only adds interventions has positive all-ADE
point estimates in every view, from +0.0074% to +1.1750%. It still has a negative
hard-subset interval, though, and the learned harm scores are not calibrated.
Both variants preserve the observed easy-case mean errors; neither is being
promoted to deployment from these development results.

The [complete comparison](outputs/publication_readiness_2026_09/european_incumbent_relative_v1/results.md),
[loss curves](outputs/publication_readiness_2026_09/european_incumbent_relative_v1/training_losses.svg),
[failure analysis](outputs/publication_readiness_2026_09/european_incumbent_relative_v1/failure_analysis.md)
and [reproduction guide](outputs/publication_readiness_2026_09/european_incumbent_relative_v1/reproducibility.md)
retain every adverse branch. All 288,000 updates completed and 379 scoped tests
pass. These are overlapping development views, not independent confirmation or
new trajectory-dynamics training. The next question is whether scene-level joint
control can make the extra interventions more reliable. Reserved sources remain
closed; image-pixel, raw-frame 8/12 results are not metric or physical-safety claims.

### Completed Fixed-Producer Study

I have completed a [fixed-producer controller study](outputs/publication_readiness_2026_09/european_fixed_producer_roles_v1/conclusions.md).
The idea is to keep the stronger trajectory forecaster and fallback unchanged,
then train the intervention controller on predictions from that same forecaster.
Separate groups of development scenes supply producer fitting, controller
supervision and readout. I trained 144 small Torch heads and 72 ridge controls,
froze every decision, and evaluated all 180 registered views.

This repairs an important problem: worst-locality easy degradation is now 0.238%
for matched supervision, compared with 7.85% for the cross-fitted-supervision
control. Both operate on identical final forecasts. But it is not a deployment
upgrade. Against the existing stopping-protected controller, all-ADE gains range
from -0.90% to +1.65%, with 12 positive and 3 negative confidence intervals.
The negative cases mainly lose useful switches the original controller made.
The next target is therefore the value of overriding the existing policy, not
relearning the whole floor-versus-neural choice.

The [full results](outputs/publication_readiness_2026_09/european_fixed_producer_roles_v1/results.md),
[loss curves](outputs/publication_readiness_2026_09/european_fixed_producer_roles_v1/training_losses.svg),
[changed-action accounting](outputs/publication_readiness_2026_09/european_fixed_producer_roles_v1/changed_action_accounting.json)
and [failure analysis](outputs/publication_readiness_2026_09/european_fixed_producer_roles_v1/failure_analysis.md)
retain the adverse branches. All 288,000 updates completed and 362 scoped tests
pass. The same opened scenes recur across views; these are not independent tests.
No new trajectory forecaster was trained, reserved sources remain closed, and
deployment is unchanged. Results use image pixels and raw-frame 8/12 prediction,
not metric units, physical safety or submission-ready world-model evidence.

### Completed Producer-Conditioned Study

I have finished a [producer-conditioned controller experiment](outputs/publication_readiness_2026_09/european_producer_conditioned_v1/conclusions.md).
I trained 108 small Torch gain/harm heads to test whether knowing which model
produced a trajectory improves the decision to use it. Global, real-tag and
placebo-tag heads have the same capacity and training budget, and their primary
comparisons use identical forecasts. All 180 development views are reported.

The result is mixed, not a deployment upgrade. The real tag changes all-ADE gain
by -0.50% to +0.94% versus the global head. The controller consistently improves
over its own two-source fallback, but that fallback can be worse than the existing
system. Worst-locality easy degradation reaches 8.03%. The error breakdown shows
that all seven easy violations already have a weak fallback; six remain violations
even after the neural controller helps. A learned risk score is not a safety
guarantee either: realized harm often exceeds the predicted budget.

The [full matrix](outputs/publication_readiness_2026_09/european_producer_conditioned_v1/results.md),
[loss curves](outputs/publication_readiness_2026_09/european_producer_conditioned_v1/training_losses.svg),
[paired comparisons](outputs/publication_readiness_2026_09/european_producer_conditioned_v1/paired_changes.svg)
and [failure analysis](outputs/publication_readiness_2026_09/european_producer_conditioned_v1/failure_analysis.md)
include the adverse branches and scenes. The 216,000 updates completed, checkpoint
replays match, and 344 scoped tests pass. These are newly trained controllers,
not newly trained trajectory forecasters or independent confirmation. Deployment
stays unchanged. This motivated the fixed-producer study above, separating
producer fitting, controller supervision and readout. These remain image-pixel,
raw-frame 8/12 development results.

### Preceding Support-Factorization Study

I completed a [support-factorization experiment](outputs/publication_readiness_2026_09/european_support_factorization_v1/conclusions.md).
It separates support for observed motion, support for model disagreement, and
the requirement that the same training sources support both. The forecasts,
learned heads and thresholds stayed fixed. All 936 registered development
comparisons are reported, including the unchanged-policy replay controls.

The result rules out a simple fix. None of the four support filters has a positive
all-ADE confidence interval against the unchanged stopping-protected controller.
The historical-motion rejection alone removes more benefit than harm in every
view under both target families. Dropping parts of the joint filter recovers
some accuracy, but recovering a loss is not a new model improvement. Matching
intervention counts within each current frame also gives no consistent ranking
advantage.

The [full results](outputs/publication_readiness_2026_09/european_support_factorization_v1/results.md),
[figure](outputs/publication_readiness_2026_09/european_support_factorization_v1/factor_changes.svg),
[failure analysis](outputs/publication_readiness_2026_09/european_support_factorization_v1/failure_analysis.md)
and [reproduction guide](outputs/publication_readiness_2026_09/european_support_factorization_v1/operation_zh.md)
retain the negative comparisons and locality failures. The worst observed
positive-easy degradation remains 0.437%, and the small stopping repair is intact.
There are 326 passing scoped tests and a complete independent arithmetic check,
but no new neural training, deployment or independent confirmation. That negative
result motivated the producer-conditioned refit above instead of more support-cutoff tuning.
These remain image-pixel, raw-frame 8/12 development results, not physical safety
or a submission-ready world model.

### Preceding Causal Abstention Study

I have finished a [causal abstention comparison](outputs/publication_readiness_2026_09/european_causal_abstention_v1/conclusions.md)
on the frozen forecasting system. I tested whether recent stopping and support
from the fitting scenes can reject harmful neural predictions. Each rule has
controls that intervene on exactly the same number of agents in the same current
frame. All 720 development views are reported; there was no new neural training.

Checking the latest observed step fixes the known stopping defect: none of the
four zero-error reference examples is harmed after this guard. But these examples
have only two future labels each, and the overall accuracy change is negligible.
The same-frame controls make identical decisions, so this is a small robustness
repair, not evidence of better joint-agent reasoning.

The broader support filter does not work as hoped. It lowers all-ADE performance
against the unchanged controller in every view, by about 0.015% to 0.128%. The
accounting shows why: it discards more useful than harmful interventions. There
are some positive-easy gains at matched intervention counts, but no robust overall
ranking advantage. The worst positive-easy degradation remains below 2%; that is
not a physical-safety certificate or independent confirmation.

The [full comparison](outputs/publication_readiness_2026_09/european_causal_abstention_v1/results.md),
[figure](outputs/publication_readiness_2026_09/european_causal_abstention_v1/guard_changes.svg),
[failure analysis](outputs/publication_readiness_2026_09/european_causal_abstention_v1/failure_analysis.md)
and [reproduction guide](outputs/publication_readiness_2026_09/european_causal_abstention_v1/operation_zh.md)
retain the tradeoffs and negative results. All decision and arithmetic checks pass,
with 314 scoped tests. These are image-pixel 8/12 results on opened development
scenes, not a new dynamics model or independent confirmation. Deployment stays
unchanged. This motivated the support-factorization study above rather than
another threshold search on the same results.

### Preceding Target-Learning Study

I have completed a [matched target-learning experiment](outputs/publication_readiness_2026_09/european_floor_relative_v1/conclusions.md).
The question was whether the controller would make better decisions if it learned
gain and harm relative to its actual strong fallback, rather than constant
velocity. I trained 234 small Torch heads with 468,000 updates, keeping causal
inputs, capacity and sampling matched. Training targets came from models that
excluded the locality being scored, not from in-sample teacher predictions.

The result is informative but negative for that particular repair. Both-target
controllers improve overall ADE over the protected fallback by 0.12% to 1.64%,
with positive conditional locality-bootstrap intervals in all 36 views. Yet the
matched CV-target control is better in 33 of those views. Twenty paired intervals
favor the control; none favor the new target. A model can beat a baseline without
its proposed change explaining the improvement.

The remaining safety issue is also more specific now. Positive-easy degradation
stays below 2%, but some interventions still harm zero-error reference cases.
The four underlying examples have stopped at the latest observed step, while
the existing guard only checks whether there was movement anywhere in the past.
They also come from one locality absent from the corresponding fitting sets.
That points toward support-aware stop/start abstention, not another round of
threshold selection on these results. The labels cover only two future steps
for those cases, so they cannot establish full-horizon safety either.

The [complete comparison](outputs/publication_readiness_2026_09/european_floor_relative_v1/results.md),
[paired figure](outputs/publication_readiness_2026_09/european_floor_relative_v1/reference_targets.svg),
[training losses](outputs/publication_readiness_2026_09/european_floor_relative_v1/training_losses.svg)
and [failure analysis](outputs/publication_readiness_2026_09/european_floor_relative_v1/failure_analysis.md)
keep every arm visible. Checkpoint checks and independent arithmetic pass, along
with 301 scoped tests. These remain opened-development, image-pixel 8/12 results,
not independent confirmation or a safe neural deployment. The trajectory
forecaster was frozen; this was controller training, not new dynamics learning.
I am keeping deployment unchanged.

The [preceding frozen-fallback diagnosis](outputs/publication_readiness_2026_09/european_floor_opportunity_v1/conclusions.md)
explains why this experiment was worth testing. Reproduction and recovery are
documented in the [Chinese operation guide](outputs/publication_readiness_2026_09/european_floor_relative_v1/operation_zh.md).

### Earlier Source Experiments

I have completed a [ranking-supervision experiment](outputs/publication_readiness_2026_09/european_ranked_hurdle_v1/conclusions.md)
following the coverage diagnosis below. I trained 36 new Torch risk heads across
three seeds, with 72,000 updates. The only change was a loss that explicitly
teaches the controller to order interventions by risk. Forecasts, model capacity,
training samples and safety limits stayed fixed.

This did not establish a safe neural advantage. Some full-policy scores improve,
but comparisons at the same intervention counts show no consistent ordering gain.
Against equally protected damping, all 18 neural all-ADE comparisons are negative;
17 conditional confidence intervals favor damping. The neural policies meet the
2% positive-easy degradation limit, yet still harm zero-error constant-velocity
cases in 12 views. I am keeping deployment unchanged.

The [complete results](outputs/publication_readiness_2026_09/european_ranked_hurdle_v1/results.md),
[matched-ranking figure](outputs/publication_readiness_2026_09/european_ranked_hurdle_v1/ranking_comparison.svg)
and [training losses](outputs/publication_readiness_2026_09/european_ranked_hurdle_v1/training_loss.svg)
retain every registered comparison. All 36 checkpoint replays, 216 metric views
and separate arithmetic checks pass; 234 scoped tests pass. These checks establish
computational reproducibility, not scientific success or independent confirmation.

Training traces identify a concrete next test: easy-event ranking receives far
fewer valid training pairs than all-event ranking. I will test pairing supported
event rows first, without changing the samples, risk limit or evaluation roles.
That may address weak supervision; it is not yet a demonstrated explanation for
the whole failure. These remain image-pixel 8/12 development results, not metric
prediction, physical safety or a new deployment model.

I have completed a [matched-coverage diagnosis](outputs/publication_readiness_2026_09/european_hurdle_coverage_v1/support_v2/conclusions.md)
to answer a question left by the last experiment: does the new risk head choose
better interventions, or mainly change how often the model intervenes?

I kept every forecast and fitted head frozen and compared the two risk rankings
at the same intervention counts in each locality. The answer depends on the
event target and scene split. There is no consistent ranking advantage. Some
accuracy gains come from making more switches; much of the easy-event protection
comes with making fewer. At the old intervention counts, the new neural ranking
breaks the 2% easy-degradation limit in six of nine comparisons. The old ranking
at the new, lower counts stays below that limit, but still harms some cases
where constant velocity was already exact.

The [complete results](outputs/publication_readiness_2026_09/european_hurdle_coverage_v1/support_v2/results.md)
and [all-comparison figure](outputs/publication_readiness_2026_09/european_hurdle_coverage_v1/support_v2/matched_ranking.svg)
retain both count anchors and the unequal-support cases. All 216 views reproduce,
with a separate arithmetic implementation and 227 scoped tests. This was a new
diagnostic, not another training run or a deployable policy. Forced-count controls
can violate their predicted-risk limit, so I am not selecting one for deployment.
These remain image-pixel 8/12 development results, not independent confirmation.
Next I will target ordering and unsupported cases separately, with the same
strong damping control and no relaxation of the safety limit.

I have completed the [occurrence-severity experiment](outputs/publication_readiness_2026_09/european_hurdle_risk_v1/conclusions.md).
Instead of asking a risk head to learn only an average error increase, I also
teach it whether a harmful switch occurs and how large that harm is when it
occurs. I trained 72 small Torch heads across three seeds, with 144,000 updates,
against an otherwise identical control. Forecasts and safety limits stayed fixed.

This helps one important failure mode: the largest positive-easy degradation
across neural views falls from 17.25% to 0.67%. It does not solve the whole
problem. The new controllers still damage some cases where constant velocity
has zero error, and none of the 18 all-ADE comparisons establishes an advantage
over equally protected damping. Two hard-subset comparisons improve slightly,
but they come from one split; most favor damping. I am not promoting the model.

The [complete results](outputs/publication_readiness_2026_09/european_hurdle_risk_v1/results.md),
[comparison figure](outputs/publication_readiness_2026_09/european_hurdle_risk_v1/objective_comparison.svg)
and [failure analysis](outputs/publication_readiness_2026_09/european_hurdle_risk_v1/failure_analysis.md)
retain the tradeoffs and the stronger controls. All 72 checkpoint replays and
144 metric views reproduce; 218 scoped tests pass. These are image-pixel 8/12
development results, not independent final-test evidence or physical safety.
This led to the matched-coverage diagnosis above. Reserved data and current
deployment remain unchanged.

I have completed a [matched cost-head experiment](outputs/publication_readiness_2026_09/european_geometric_cost_v1/conclusions.md).
The question is whether a controller makes better decisions when its predicted
gain and harm are constrained by how far a candidate forecast moves from the
causal baseline. I kept the forecasts and risk limits fixed and trained 54
small neural heads across three seeds, with 108,000 optimizer updates in total.

The answer is mixed, and not yet safe enough. The new risk heads allow more
useful neural predictions through. Six new comparisons have a positive ADE
interval against equally protected damping, but all six fail the safety checks.
The strongest positive comparison has 17.64% worst-scene easy degradation.
Bounding the size of a predicted cost does not make that cost well calibrated
on the samples the controller chooses to change.

The [complete comparison](outputs/publication_readiness_2026_09/european_geometric_cost_v1/head_ablation.svg),
[training losses](outputs/publication_readiness_2026_09/european_geometric_cost_v1/training_loss.svg)
and [failure analysis](outputs/publication_readiness_2026_09/european_geometric_cost_v1/failure_analysis.md)
retain every arm, including the stronger damping controls. All 54 checkpoints
and 144 policy views reproduce; 210 scoped tests pass. I am not changing
deployment or claiming a successful neural dynamics model. These are
opened-source, image-pixel 8/12 development results, not independent final-test
evidence. My next experiment will separate event support from conditional harm
rather than relax the safety limit to make the current results look better.

I have completed the [frozen producer-transport diagnostic](outputs/publication_readiness_2026_09/european_producer_transport_v1/conclusions.md).
It tests a specific explanation for the controller's failures: the gain/risk
head learns from smaller forecasting models, then controls a different final
forecaster. I kept the heads, risk limits and excluded evaluation rows fixed,
and replaced the final predictor with each of its two smaller counterparts.

That replacement is not a reliable repair. Only 9 of 36 controlled ADE
comparisons improve; five conditional intervals favor replacement and fifteen
favor the original predictor. The smaller models still miss substantial harm
on selected samples. Some changes help, but their effects depend on the fitting
scenes and do not establish a consistent safe neural advantage.

The [all-comparison figure](outputs/publication_readiness_2026_09/european_producer_transport_v1/producer_comparison.svg),
[raw trajectory results](outputs/publication_readiness_2026_09/european_producer_transport_v1/raw_forecast_results.md)
and [failure taxonomy](outputs/publication_readiness_2026_09/european_producer_transport_v1/failure_taxonomy.md)
separate forecast quality from intervention errors. The experiment generated
18 inference banks from frozen checkpoints, not new training. All 72 views
reproduce and 201 scoped tests pass. Deployment and reserved data stay unchanged.
These are opened-source, image-pixel 8/12 development results, not independent
confirmation. My next repair targets candidate-specific gain and harm learning,
not choosing a favorable producer after seeing its evaluation results.

I have completed the [nested source-calibration study](outputs/publication_readiness_2026_09/european_nested_calibration_v1/conclusions.md).
I trained 18 inner forecasting models and 54 gain/risk heads with the calibration
scenes excluded from their complete training chain. Three seeds, both predefined
scene-role rotations and all 72 policy views are retained.

Calibration helps some safety checks, but the neural model still does not beat
equally protected causal damping. All 36 direct ADE comparisons favor damping;
34 conditional intervals are strictly negative. Neural observed safety improves
from 2/12 views without calibration to 5/12 under each calibration method,
compared with 12/12 and 11/12 for the matched damping controls.

The distinction matters: a rule can satisfy the risk limits on its calibration
scenes and still fail on another scene. The [all-view figure](outputs/publication_readiness_2026_09/european_nested_calibration_v1/calibration_comparison.svg)
and [calibration-to-readout audit](outputs/publication_readiness_2026_09/european_nested_calibration_v1/calibration_transport.md)
show that gap. All new checkpoints, decisions and metrics reproduce; 196 scoped
tests pass. I am not promoting a new model or claiming independent risk control.
These remain opened-source, image-pixel 8/12 development results. Next I will
separate producer-training shift from prediction and ranking errors before
committing to another model change.

I have completed the [symmetric-risk follow-up](outputs/publication_readiness_2026_09/european_symmetric_risk_v1/conclusions.md).
Keeping the forecasts and utility heads fixed, I trained 36 new risk heads
with a symmetric loss. Neural ADE gain over constant velocity rises to
1.13%, 1.58% and 2.09% across three seeds, but worst-locality easy degradation
rises to 2.46%, 7.29% and 16.71%. All exceed the 2% limit.

The result clarifies a real tradeoff: the earlier conservative risk estimate
blocked some useful predictions, but also prevented genuine harm. The new heads
underestimate harm specifically on the samples they choose to change. They do
not establish a stable accuracy advantage over equally protected causal damping,
so I am not changing deployment or claiming a safe neural dynamics model.

The [all-view figure](outputs/publication_readiness_2026_09/european_symmetric_risk_v1/risk_ablation.svg),
[complete comparisons](outputs/publication_readiness_2026_09/european_symmetric_risk_v1/results.md)
and [risk reliability tables](outputs/publication_readiness_2026_09/european_symmetric_risk_v1/risk_reliability_table.md)
retain both the gains and failures. All 36 checkpoints and full metrics reproduce;
185 scoped tests pass. These are source-development pixel-space 8/12 results,
not independent confirmation. The next question is whether risk can be calibrated
on the selected samples without losing the useful neural interventions.

I have now run the [symmetric-utility experiment](outputs/publication_readiness_2026_09/european_symmetric_utility_v1/conclusions.md).
It changes one factor: the utility head estimates gain and harm with symmetric
MSE, while every risk head and risk limit stays frozen. I trained 18 new heads
across three seeds and retained all 48 policy comparisons. This substantially
reduces harm-estimation bias, but it does not establish a neural dynamics advantage.

In the fixed easy-event neural-risk views, neural ADE gain over CV is now
0.27%, 0.20% and 0.42%; the equally protected damping control reaches
2.09%, 2.20% and 2.16%. All 24 direct neural-versus-damping intervals still favor
damping. Some other views fail easy or zero-error preservation, and one joint
solver call safely falls back. I keep these negative results in the
[complete comparison](outputs/publication_readiness_2026_09/european_symmetric_utility_v1/results.md).
The [all-view figure](outputs/publication_readiness_2026_09/european_symmetric_utility_v1/utility_ablation.svg)
shows why a better cost estimate is not yet a better neural controller.

All new checkpoints and complete metrics reproduce, and 179 scoped tests pass.
These remain opened-source development results, not independent calibration
or confirmation. I am not changing deployment. The next question is how much
of the remaining limitation comes from risk estimation versus neural errors
that the available history cannot reliably distinguish.

I have completed the [opportunity diagnosis](outputs/publication_readiness_2026_09/european_opportunity_diagnosis_v1/conclusions.md).
The neural candidate has more hindsight opportunity than fixed damping, but its
protected controller captures much less of it. In the easy-event neural-risk
views, about 59--61% of attainable neural benefit is rejected first by utility
scoring; most remaining misses occur at the risk gate. The controller captures
only 1.04--2.78% of gross neural opportunity, versus 23.31--25.34% for damping.
This does not mean the hindsight gains are learnable from past observations.

The diagnostic reproduces all 48 frozen policies without fitting a model or
opening reserved data. It points to a specific next experiment: the utility
head currently uses a conservative harm loss, then feeds a second conservative
risk gate. That motivated the symmetric-utility experiment above, with risk
heads and limits unchanged, not a relaxed threshold selected from these
outcomes. The [attribution figure](outputs/publication_readiness_2026_09/european_opportunity_diagnosis_v1/opportunity_attribution.svg)
and [all tables](outputs/publication_readiness_2026_09/european_opportunity_diagnosis_v1/tables.md)
keep the missed opportunities, harms and producer differences visible.

I have completed the [matched risk-protected motion study](outputs/publication_readiness_2026_09/european_protected_motion_v1/conclusions.md).
The result challenges the current neural-trajectory hypothesis: when fixed
damping receives the same gain/harm learning and risk rules, it outperforms
the neural candidate in all 24 paired pointwise comparisons. Each conditional
locality-bootstrap interval favors protected damping. Both candidates use the
same risk budget, but they need not intervene on the same number of agents.

With easy-event neural risk heads, protected damping improves ADE over CV by
1.81--1.94% across three seeds, versus 0.17--0.43% for the neural trajectories.
This suggests useful intervention learning, not yet a neural dynamics advantage.
Other controls still fail local easy preservation, joint decisions lack a stable
matched-count benefit, and independent calibration remains untested. I am not
changing deployment or claiming submission readiness.

The experiment fitted 45 new control heads, including 54,000 real Torch updates,
and verified 45 existing neural-candidate heads. All 48 policy views are retained;
all 90 checkpoints reproduce sampled predictions exactly. The
[full comparison](outputs/publication_readiness_2026_09/european_protected_motion_v1/results.md),
[contrast figure](outputs/publication_readiness_2026_09/european_protected_motion_v1/candidate_contrasts.svg)
and [operation guide](outputs/publication_readiness_2026_09/european_protected_motion_v1/operation_zh.md)
record the result and its limits. My next step is to separate forecast quality,
risk-estimation error and producer-training shift on the opened source scenes,
not to tune on reserved outcomes or simply increase model size.

I have completed the [event-conditional risk study](outputs/publication_readiness_2026_09/european_conditional_risk_v1/conclusions.md):
36 fitted risk heads and 24 fixed policy views, with the forecast bank unchanged.
The new easy-event neural heads reduce worst-locality easy degradation to
0--0.82% and harm none of the four observed zero-error CV cases in the full
pointwise evaluation. But their ADE gain over CV falls to 0.17--0.43%, well
below the preceding policy and the strong fixed-damping accuracy control.
This is a protection/utility tradeoff, not a new deployment result.

The matched controls show that easy-event targets help within the neural risk
model, but a simple ridge risk head remains competitive. Joint decisions still
lack a stable same-intervention-count advantage, and two unguarded neural seeds
exceed the 2% worst-locality easy limit on the joint pilot. I retain every seed
and negative control in the [results](outputs/publication_readiness_2026_09/european_conditional_risk_v1/results.md)
and [failure analysis](outputs/publication_readiness_2026_09/european_conditional_risk_v1/failure_analysis.md).
All 36 checkpoints reproduce sampled predictions exactly; 152 tests pass in
the completion scope. Independent selection, calibration and confirmation stay
closed. The protected-motion comparison above now completes that follow-up.

The [tradeoff figure](outputs/publication_readiness_2026_09/european_conditional_risk_v1/risk_utility_tradeoff.svg)
and [operation guide](outputs/publication_readiness_2026_09/european_conditional_risk_v1/operation_zh.md)
make the result reproducible. These are source-development image-pixel 8/12
results, not historical t50, calibrated physical safety or independent proof.

I have completed the [CV-reference repair](outputs/publication_readiness_2026_09/european_cv_reference_v1/conclusions.md):
nine ridge and nine neural cost heads, with the original forecasts frozen.
Changing the fallback and its gain/harm supervision to causal constant velocity
reduces the neural policy's mean easy-case degradation from 12.81--13.38% to
1.90--2.38%. Its ADE gain over CV is 4.18--4.43% across three seeds, but the
advantage over fixed damping 0.97 remains uncertain. This is a partial repair,
not a deployment result.

The remaining failure is specific: worst-locality easy degradation is still
8.71--12.54%, and each seed harms one of the four zero-error CV cases. Those
four cases have only two observed future labels and come from a held source
fold with no fitting examples of that event. I keep them in the evaluation;
removing difficult safety cases would not solve the problem. The joint-control
pilot contains none of them and cannot validate zero-event protection.
[All controls and intervals](outputs/publication_readiness_2026_09/european_cv_reference_v1/results.md),
[failure analysis](outputs/publication_readiness_2026_09/european_cv_reference_v1/failure_analysis.md)
and the [repair figure](outputs/publication_readiness_2026_09/european_cv_reference_v1/repair_contrasts.svg)
are retained. All 18 cost checkpoints reproduce sampled predictions exactly;
145 scoped tests pass. Independent data roles and deployment remain unchanged.

I have completed the [European Squares source-only experiment](outputs/publication_readiness_2026_09/european_source_forecast_v1/conclusions.md):
18 real Transformer fits, three seeds and 72,000 optimizer updates. Mean-seed
ADE improves by 4.11% over the baseline selected on other fitting localities,
with a conditional locality-bootstrap interval of [1.37%, 6.95%]. But easy-case
degradation is 13.65--14.39%, and every seed harms the four zero-error CV cases.
This is a useful prediction signal, not a safe deployable model.

The comparison needs care: fixed damping 0.97 achieves 3.98% gain over CV,
versus 2.16% for the neural predictor on the same scale. I cannot claim that
the neural model beats every strong baseline. The training-selected fallback
also fails easy preservation; a fallback is not automatically a safety floor.

The [nested gain/harm study](outputs/publication_readiness_2026_09/european_source_intervention_v1/conclusions.md)
adds nine ridge and nine neural cost heads. Joint intervention does not show a
stable advantage over independent decisions at the same intervention count.
I am retaining these negative controls and changing neither deployment nor the
risk limits. The [audit entry](outputs/publication_readiness_2026_09/audit_entry_20260924.md)
links results, losses, checkpoint replays and limitations. All 36 checkpoint
checks reproduce sampled predictions exactly; 127 scoped tests pass. Reserved
selection, calibration and confirmation roles, including DroneCrowd, remain closed.

I have frozen [locality-level roles for European Squares](outputs/publication_readiness_2026_09/european_squares_roles_v1/conclusions.md)
before looking at prediction errors: 12 training groups, 6 for model selection,
12 for risk calibration and 6 for confirmation. All recordings from a locality
stay together. Only the training groups are open. The first registered source
cohort is built from all 163 training recordings, with 318,969 prediction targets
and incomplete-history neighbors retained. It
[reproduces exactly from the raw recordings](outputs/publication_readiness_2026_09/european_squares_source_v1/conclusions.md);
489 future-truncation checks and 79 scoped tests pass. Future labels are stored
separately from past-only model inputs; missing futures do not remove targets.

The [full raw intake](outputs/publication_readiness_2026_09/european_squares_intake_v2/conclusions.md)
covers 152,372,066 rows in 376 recordings. The new
[partial-clip screen](outputs/publication_readiness_2026_09/european_squares_overlap_v1/conclusions.md)
finds no exact eight-frame dynamic matches or cross-locality integer-pixel matches.
Near-static quantized matches remain within their locality. The
[prior-exposure search](outputs/publication_readiness_2026_09/european_squares_exposure_v1/conclusions.md)
is explicitly bounded; it does not prove universal independence or locate
unknown remote assets. These are source-data improvements, not new model scores.
The task uses released detector tracks in image pixels and raw frame indices,
not verified online ground truth, meters or seconds. Reserved prediction errors
and DroneCrowd confirmation remain closed; deployment is unchanged.

I have completed the [causal-neighborhood and annotation audit](outputs/publication_readiness_2026_09/moving_zero_support_v1/conclusions.md)
following the failed zero-reference guard. All 36 views reproduce exactly. The
seven moving zero-CV windows come from only three tracks; none has a matching
zero-event label among its nearest 512 effective source neighbors in either
tested feature bank. Ordinary similar histories are present, so this is a
relevant-event support gap, not proof that prediction is impossible.

Raw annotations also expose a limit to the causal claim: six of the seven
histories include generated points bracketed by later controls beyond the query.
That is a provenance concern, not proof of a particular interpolation algorithm.
The fixed sampled labels are correct, but strict online sensor-as-of causality
is not established. I am keeping the evaluation grid, risk limits and deployment
unchanged. The next priority is admissible independent trajectory support and
annotation-time provenance, not another threshold sweep on these same tracks.
[Tables and figure](outputs/publication_readiness_2026_09/moving_zero_support_v1/tables.md),
[execution record](outputs/publication_readiness_2026_09/moving_zero_support_v1/execution_notes.md).

I have completed the [explicit zero-reference risk experiment](outputs/publication_readiness_2026_09/zero_atom_v1/conclusions.md).
Separately estimating when constant velocity is exactly correct does not repair
protection: Transformer ADE gain falls from 3.60% to 2.97%, with the same five
harmed window/seed cases; EqMotion gain falls from 3.34% to 2.80%, with harms
increasing from six to eight after joint reallocation. Matched-intervention
controls also outperform the added guard. I am retaining this negative result,
not changing deployment.

The [support audit](outputs/publication_readiness_2026_09/zero_atom_v1/fit_support.md)
finds only 2--7 relevant moving zero-reference training windows per readout,
despite thousands of stopped examples. An empirical probability of zero is not
a safety guarantee. The [full controls and paired intervals](outputs/publication_readiness_2026_09/zero_atom_v1/results.md)
and [figure](outputs/publication_readiness_2026_09/zero_atom_v1/risk_tradeoff.svg)
show the utility/protection tradeoff. These are 36 new leaf readouts on frozen
forests, not new neural forecasts; the four source sites remain development-exposed.

I have completed the [cutoff-relative risk study](outputs/publication_readiness_2026_09/cutoff_relative_risk_v1/conclusions.md):
36 fresh fits with the same predictors, targets, source draws and risk limit.
Restoring motion scale relative to the training error cutoff reduces the preceding
dimensionless population policy's worst easy degradation from 4.45% to 1.27% for
Transformer and 9.01% to 1.46% for EqMotion. But their average ADE gains over
constant velocity fall to 3.60% and 3.34%, close to the native-feature controls.
The tiny advantages over those controls remain uncertain, and some zero-error
baseline cases are still harmed. I am not changing deployment.

The [full table](outputs/publication_readiness_2026_09/cutoff_relative_risk_v1/results.md),
[training losses](outputs/publication_readiness_2026_09/cutoff_relative_risk_v1/training_losses.md)
and [tradeoff figure](outputs/publication_readiness_2026_09/cutoff_relative_risk_v1/risk_tradeoff.svg)
retain all controls. This is a partial risk-representation repair on four
development-exposed SDD sites, not new neural dynamics, independent confirmation
or a calibrated safety guarantee.

I have completed a [matched dimensionless risk-head study](outputs/publication_readiness_2026_09/dimensionless_risk_v1/conclusions.md):
72 fresh fits with unchanged predictors, matched source draws and three seeds.
Removing explicit native-unit features increases the population policy's average
ADE gain over constant velocity from 3.57% to 4.99% for Transformer and 3.30% to
5.97% for EqMotion. But their worst easy-case degradation rises to 4.45% and 9.01%,
above the 2% ceiling. Better average prediction is not enough to justify deployment.

A more restrictive Transformer control gives 2.99% ADE gain, 0.94% worst
positive-easy degradation and no observed zero-CV harms. Its advantage over the
old strict control remains uncertain across physical sites, so I am retaining
it as a research signal, not selecting a new deployable winner. The
[tradeoff figure](outputs/publication_readiness_2026_09/dimensionless_risk_v1/risk_tradeoff.svg),
[all controls](outputs/publication_readiness_2026_09/dimensionless_risk_v1/results.md)
and [training losses](outputs/publication_readiness_2026_09/dimensionless_risk_v1/training_losses.md)
show both gains and failures. These are four-site development results under the
native 8/12 protocol, not external confirmation, historical t50 or a safety guarantee.

I have found a [coordinate-unit dependency in the frozen risk heads](outputs/publication_readiness_2026_09/imptc_input_contract_v1/conclusions.md).
Holding the motion and normalized predictions fixed, changing only two native-unit
features flips the Transformer head's predicted signed easy-risk sign in 369 of 754
diagnostic windows. I have added a separate unit-free input contract. Its initial
three EqMotion numerical failures are retained, and a [versioned precision repair](outputs/publication_readiness_2026_09/imptc_precision_v2/conclusions.md)
now passes the same probes; predictive value still requires source-only training.
This is an input-mechanism result, not forecasting
improvement: no external prediction errors were opened, no model was retrained,
and deployment is unchanged. The run and replay are verified; DroneCrowd remains
closed for confirmation.

I have added a [verified IMPTC source adapter](outputs/publication_readiness_2026_09/imptc_intake_v1/conclusions.md)
to work toward independent-site evidence. The official sample package contains
142,361 observations across four recordings, but only one physical intersection
and 61 person-labelled tracks. Every converted row and history-support count has
been replayed and checked. No forecast errors have been opened and no model has
been fitted on it. Source-processing provenance and related-site exposure still
need resolving before admission; this is not independent calibration or a new
model result. DroneCrowd confirmation stays closed.

I have completed a [fixed comparison of shared risk budgets](outputs/publication_readiness_2026_09/risk_subsidy_v1/conclusions.md).
It separates credit from other agents' predicted improvements from budget
contributed by agents whose forecasts stay unchanged. Removing both eliminates
the observed zero-error-baseline harms, but reduces Transformer ADE gain from
2.94% to 1.40% and EqMotion from 2.83% to 1.30%. The restricted rules also trail
the old strict control. I am retaining this as a mechanism result, not promoting
a new policy or claiming safety.

The comparison keeps every fitted model and the 2% ceiling unchanged, includes
all 175,756 source windows, and reports all 33 controls. Equal-intervention
comparisons show that the restrictions change which useful targets are admitted,
not just the intervention rate. Unknown futures and numerical limitations remain
explicit in the [results](outputs/publication_readiness_2026_09/risk_subsidy_v1/results.md).
These are development-exposed native8/12 results, not historical t+50 or independent
confirmation. The [method positioning note](outputs/publication_readiness_2026_09/risk_subsidy_v1/literature_and_claim_limits.md)
explains why a predicted-risk constraint is not a statistical safety guarantee.
Deployment is unchanged and reserved confirmation data remain closed.

I have completed the [net easy-risk study](outputs/publication_readiness_2026_09/net_easy_moment_guarded_v1/conclusions.md):
36 new risk heads on unchanged forecasts, with three seeds and all registered
source windows. Accounting for both improvement and harm recovers average ADE
gains of 2.94% for Transformer and 2.83% for EqMotion over constant velocity.
Transformer's improvement over the previous strict rule remains uncertain;
EqMotion has a positive nominal development contrast, but worse easy-case
performance. Both policies harm some zero-error baseline cases, so I am not
changing deployment or claiming a safety guarantee.

These are four-site, development-exposed results under the 8-observed/12-predicted
annotation-step protocol, not the historical t+50 scores or independent
confirmation. The [full comparisons](outputs/publication_readiness_2026_09/net_easy_moment_guarded_v1/results.md),
[training losses](outputs/publication_readiness_2026_09/net_easy_moment_guarded_v1/training_losses.md)
and [method note](outputs/publication_readiness_2026_09/net_easy_moment_guarded_v1/method_note.md)
retain the negative controls and explain why net-risk accounting is a weaker
constraint. Independent confirmation remains closed.

I have completed the [same-query risk allocation study](outputs/publication_readiness_2026_09/easy_allocation_risk_scaled_v1/conclusions.md)
on the full registered source population. Pooling risk across targets recovers
some useful intervention: Transformer ADE gain reaches 1.280%, with 0.112%
worst-site/seed easy degradation. It still trails the previous strict rule's
2.437% gain, and the nonadditive interaction term adds almost nothing beyond
the matched unary control. I am not promoting this as a new best policy.
The study also exposed and repaired a small-risk numerical solver issue without
loosening the risk limit; both versions and all negative results are retained.
Full replay, separate arithmetic and 115 scoped tests pass. No predictor was
retrained, no threshold was tuned, and independent confirmation remains closed.

I have completed the [conditional easy-risk experiment](outputs/publication_readiness_2026_09/easy_moment_v1/conclusions.md):
36 new risk heads on frozen damping, Transformer and EqMotion forecasts. Directly
learning easy-weighted harm protects the observed easy cases but rejects nearly
all useful neural intervention. Transformer ADE gain falls from 2.437% under
the previous strict rule to 0.013%; EqMotion falls from 1.609% to 0.001%.
Equal-intervention comparisons also favor the simpler product-of-marginals
control. I am retaining this as a negative result, not replacing the current
policy. The complete run, losses, source exclusions and independent arithmetic
checks are documented; these remain four-site development results, not external
confirmation or a safety guarantee.

I have acquired and audited the [official HT21/CroHD annotations](outputs/publication_readiness_2026_09/ht21_annotations_v1/conclusions.md)
to investigate denser external interactions. All 1,188,496 released GT rows and
the history-support counts reproduce, but I have not admitted them as a new
forecasting benchmark: three of four labelled recordings report camera motion,
the annotations include interpolation, and independent physical sites remain
unverified. The input reader preserves static people without using their
whole-video motion label. This is a verified data asset, not a new model result
or independent calibration. DroneCrowd remains closed for confirmation.

I have extended the protected-motion comparison to
[full EqMotion forecasts](outputs/publication_readiness_2026_09/protected_eqmotion_controls_v1/conclusions.md),
reusing the existing excluded-site predictors and fitting the missing matched
forest heads. Strict neural protection gives 1.61% average ADE gain with 0.45%
worst-site/seed easy degradation. At equal intervention counts, protected simple
damping gives 2.67% gain and 1.62% easy degradation, although their paired
difference remains uncertain. This has narrowed my claim: learning when to
intervene is useful, but I have not shown that neural forecasting is indispensable.
The comparison is replay-verified development evidence, not an independent test
or a new deployment decision. Losses, complete controls and negative results are
in the report; reserved confirmation data remain closed.

I have now completed a [matched comparison against protected simple motion](outputs/publication_readiness_2026_09/protected_motion_controls_v1/conclusions.md),
with 156 new control-head fits and twelve verified existing heads. Giving damping
the same learned protection improves its average source result beyond the
Transformer, but damages easy cases beyond the specified ceiling. With forest
protection, their average difference is uncertain. Transformer retains a useful
observed gain/easy-degradation tradeoff, but I cannot yet claim that neural
forecasting is indispensable. These are four-site development results, not a new
external test or deployment decision. All actions, losses and negative results
remain in the report; DroneCrowd confirmation remains closed.

I have completed the [full DUT frozen-model readout](outputs/publication_readiness_2026_09/dut_frozen_readout_v1/conclusions.md):
27 recordings, two locations and 420,364 past-eligible target windows, including
those with incomplete future labels. This is a fixed external diagnostic, not
a quick sample or a new model-selection round. Protected neural policies improve
average ADE over constant velocity by about 1.6% and 1.7% across three seeds for
Transformer and EqMotion respectively, with no observed degradation in the small
predefined easy subset. Unprotected models gain more on average but harm easy cases.

The limitations matter: a damped-motion control has better overall ADE but fails
easy preservation, and joint selection adds almost nothing beyond unary selection.
Only two sites and 922 easy targets support this comparison, so it is not a safety
guarantee or a confirmed world-model contribution. All fixed views remain reported;
no external winner is deployed. Independent raw-row coverage and aggregate replay
pass. A fresh process also reproduces all twelve views on 6,715 fixed query times
across 66 registered chunks from all 27 recordings. This is chunk verification,
not a second full run. DroneCrowd confirmation remains closed.

I have completed [six fixed source-only predictor fits](outputs/publication_readiness_2026_09/external_predictor_refit_v1/conclusions.md):
the existing Transformer and EqMotion control, each with three seeds on the
already-used SDD source sites. All 24,000 registered updates completed locally in
74.6 minutes of fitting time. Checkpoint reloads and fixed-input predictions
reproduce exactly; the [training losses](outputs/publication_readiness_2026_09/external_predictor_refit_v1/training_losses.md)
are available rather than just a completion status. These are trained predictors,
not external validation results. The [past-only input interface](outputs/publication_readiness_2026_09/external_prefix_adapter_v1.md)
keeps observation construction separate from future labels.

I have also completed the [gain/harm estimation bank](outputs/publication_readiness_2026_09/external_cost_bank_v1/conclusions.md)
for these predictors: six neural heads and six sampling-matched tree controls.
Their cost targets come from models that excluded each row's source site, not
from the new predictors' in-sample errors. All twelve checkpoints reproduce their
fixed-input scores; the [loss records and limitations](outputs/publication_readiness_2026_09/external_cost_bank_v1/training_losses.md)
are available. This is source-only fitting, not held-out evaluation of the complete
policy. I have now frozen and replayed the [complete inference chain](outputs/publication_readiness_2026_09/external_policy_chain_v1/conclusions.md),
including the joint-choice controls and fallback rules. All twelve views reproduce
on 99 fixed source-scene queries. Joint selection changes no decisions in those
probes, so this step does not establish an interaction contribution. Independent
external calibration and confirmation remain unfinished, and deployment is unchanged.

I am prioritizing independent external scenes for the next validation study.
The [DroneCrowd annotation audit](outputs/publication_readiness_2026_09/dronecrowd_annotations_v1/conclusions.md)
now covers the complete official annotation archive: 112 clips, 20,800 tracks
and 4.86 million visible records. There is enough structural history support,
but clip IDs do not establish independent physical scenes. The audit also finds
coordinate differences between supplied formats and missing interpolation
provenance. I have not trained on these annotations. I now reserve the entire
collection for candidate external confirmation, without claiming that its
physical sites are already independent. Existing SDD sites remain development data;
I will not reuse them as untouched confirmation.

I have now [audited 336 official images](outputs/publication_readiness_2026_09/dronecrowd_image_audit_v1/conclusions.md),
three per clip, without downloading the full image archives. Background matching
finds a shared road scene across the supplied train/test folders and apparent
camera motion in many clips. The 68 automatic overlap groups are not verified
independent sites. The initial grouping has since been extended:
The [extended multi-view audit](outputs/publication_readiness_2026_09/dronecrowd_grouping_v2/conclusions.md)
has now checked all 6,216 recording pairs using three views, finding 66 stronger
and 46 ambiguous associations. Keeping these and the visual-review constraints
produces 45 exclusion groups, five of which cross the supplied train/test split.
I keep the entire collection in one reserved role, rather than relying on these
groups as certified independent locations. No forecast result is claimed from
this source audit.

The annotations now also have a [lossless lazy reader](outputs/publication_readiness_2026_09/dronecrowd_recordings_v1/conclusions.md):
112 compact recording caches replace repeated XML parsing without storing millions
of expanded episodes. Every array matches the source, and past inputs stay separate
from future supervision. The source caches retain their original quarantine
status; a separate [frozen reservation and admission guard](outputs/publication_readiness_2026_09/external_role_reservations_v1/conclusions.md)
prevents training or threshold selection on DroneCrowd. DUT is reserved for
calibration, with one duplicate-annotation clip excluded. These reservations
protect future evaluation; they do not grant predictive access or solve the
limited number of independent calibration sites.

The annotation audit also found a practical input-construction risk: filtering
observed agents by future label availability would discard agents with complete
past histories in about 72% of the stride-1 eight-to-twelve queries. I have added a
[past-input / future-label separation check](outputs/publication_readiness_2026_09/dronecrowd_window_separation_v1/conclusions.md)
and verified it across all 112 clips. This is a data-pipeline check, not evidence
of better forecasts or independent scene coverage.

I am also separating empirical easy-case protection from statistical risk claims.
A [source-checked calibration diagnostic](outputs/publication_readiness_2026_09/scene_risk_bound_feasibility_v1/conclusions.md)
implements an existing, tighter bounded-risk method without changing the frozen
experiments. It improves the feasibility calculation, not the model: independent
scenes are still required, and the 2% easy-error criterion is not a certificate.

The earlier [read-only CREATE connection](outputs/publication_readiness_2026_09/create_readonly_handoff_20260923.md)
worked, but the [latest refresh](outputs/publication_readiness_2026_09/create_readonly_refresh_20260924.md)
failed authentication, so I cannot report a current queue state. No M3W job was
submitted, and the separate simulation workload is untouched. The matched-control
experiments completed locally; M3W's remote project directory remains unidentified.

I have added a [portable reproduction draft](outputs/publication_readiness_2026_09/blinded_reproduction_v1/conclusions.md)
for the current evidence tables. It runs from an extracted archive with Python's
standard library, without my workspace, raw data or model weights. The isolated
run reproduces all seven result files and preserves the negative comparisons.
Direct identifying details are removed, but this is not certified anonymous or
a full model-retraining package. It improves reproducibility, not model accuracy;
independent calibration and confirmation are still missing.

The [revised evidence draft](outputs/publication_readiness_2026_09/evidence_manuscript_v2/manuscript.md)
brings the neural-versus-tree risk experiments and joint-support diagnosis into
one account. It keeps the failed primary comparisons alongside the positive
same-count result, with worst-scene/seed easy damage rather than only an average.
The [tables and figures](outputs/publication_readiness_2026_09/evidence_manuscript_v2/tables.md)
can be regenerated from sixteen pinned reports; the
[Chinese reproduction guide](outputs/publication_readiness_2026_09/evidence_manuscript_v2/reproduction_zh.md)
separates paper reconstruction, checkpoint replay and independent validation.
This is an updated research manuscript, not a new model result or a claim that
the paper is ready to submit. Independent calibration and a supported method
contribution are still the most important missing evidence.

I also checked whether the protected forecasts leave meaningful joint-agent
choices. Exhaustive comparison finds only three changed forest-policy decisions,
all from one recording/frame repeated over three seeds. The neural pools change
a few other frames, but this is still very sparse support for a joint mechanism.
This [causal-only diagnosis](outputs/publication_readiness_2026_09/protected_joint_support_v1/conclusions.md)
uses no future outcomes: improving the designed interaction objective is not
evidence of better trajectory prediction. Independent scene support remains more
important than another broad interaction-weight search.

I have completed a controlled test of whether the neural risk head's training
loss explains its gap to a simple tree comparator. Twelve new fits improve the
fitting objective in nine views, but do not repair held-scene risk ranking. The
fixed policy gains 3.45% ADE over constant velocity versus 3.40% previously; their
paired difference interval crosses zero. At equal switch counts, the new head
falls to 2.60%, versus 2.81% for the old head and 3.53% for the forest. Learning
the average cost better has not made the selected cases reliably safer. The
[full result and verification](outputs/publication_readiness_2026_09/fraction_square_v1/conclusions.md)
retain the failed primary gate and unchanged deployment. These are development
results, not independent confirmation or submission readiness.

My latest [fixed-count comparison](outputs/publication_readiness_2026_09/risk_ranking_v1/conclusions.md)
separates risk ranking from simply making fewer switches. With the same forecasts
and exactly the same switch counts, relative-risk ranking protects observed easy
cases for both the neural and tree heads. The tree retains 3.53% ADE gain over
constant velocity; the neural head retains 2.81%. Their paired difference is
0.72 percentage points, with a development-scene interval of [0.61, 0.83]. Ranking
by net gain improves average accuracy but breaks the worst-scene easy limit for
both heads. This narrows the problem to useful risk ordering as well as score
calibration, not just model size. It does not establish independent safety or a
new neural contribution; the earlier primary gate remains failed and deployment
is unchanged.

My latest comparison uses ordinary tree regression to estimate intervention costs
from the same forecasts and causal features. All 24 fits are complete. The gradual
policy gains 3.53% ADE over constant velocity, versus 3.40% with the neural risk
head. It passes the observed easy-case ceiling in every scene and seed and avoids
the neural head's exact-baseline failure. However, the paired accuracy difference
interval crosses zero, and a same-count neural ranking has higher gain but worse
easy protection. This is a useful conventional comparator, not a new world-model
contribution or a deployment. The [fixed results and risk diagnosis](outputs/publication_readiness_2026_09/forest_cost_v1/conclusions.md)
show why a more complex risk head must earn its place. Independent calibration
and confirmation are still missing, and I have not changed the deployed policy.

I have brought the completed development experiments into one
[English manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v1/manuscript.md),
with [reproducible tables and a figure](outputs/publication_readiness_2026_09/evidence_manuscript_v1/tables.md).
The central result is mixed: neural predictors improve average motion error,
but learning when to use them has not yet met the scene-wise protection target.
The earlier conservative policy gains 4.10% in average site-relative ADE, while
easy errors still rise beyond 2% in two sites. These are explored SDD development
results, not independent test evidence or a new deployment.

The manuscript keeps the failed equal-count and joint-decision controls visible.
Its tables are reconstructed from fixed result files, not another training run.
Independent calibration and confirmation remain the next scientific requirements.

One controlled experiment tested whether the risk head was trained on an
outdated set of switching decisions. I retrained twelve heads with the same
budget, updating that training emphasis every 500 steps. ADE gain was 4.03%
versus 4.10% for fixed emphasis; the paired difference interval includes zero.
Easy protection still fails in deathCircle and one gates seed. This does not
support adopting the repair: learning the selected training cases better has
not made switching reliable on another scene. I keep the negative result and
the existing deployment unchanged. The manuscript above is the preceding fixed
evidence snapshot; the [new experiment and reproducible results](outputs/publication_readiness_2026_09/adaptive_region_cost_v1/conclusions.md)
are reported separately, without changing its original comparisons.

I then tested whether two existing risk heads could review that policy's proposed
switches. The review meets the observed easy-case ceiling in every scene and seed,
but retains only 0.83% ADE gain. At exactly the same number of switches, the original
head's ranking retains 3.05%, although it still fails easy protection. This is a
real tradeoff, not a new deployable winner. A closer audit shows that the review
rejects many useful forecasts and still underestimates harm among the decisions
it accepts. Simply taking the largest risk estimate is not reliable calibration.
I have kept the [fixed comparison, negative result and veto diagnosis](outputs/publication_readiness_2026_09/cross_objective_review_v1/conclusions.md)
separate from the earlier manuscript snapshot. Independent confirmation is still
missing, and no deployment has changed.

My latest controlled test changes the risk head's loss rather than its architecture.
Twelve fresh fits raise ADE gain from 4.10% to 4.19%, but the difference interval
includes zero and easy errors still exceed the ceiling in deathCircle and gates.
The head estimates more harm on the old decisions, yet still underestimates the
decisions it now selects. A fixed turnover audit shows that harmful decisions
retained by both versions are part of the problem, not just new switches. I keep
the [verified experiment and negative finding](outputs/publication_readiness_2026_09/log_cost_v1/conclusions.md)
as development evidence; there is no new deployment or calibration claim.

The latest failure audit points to a more specific mismatch: the risk head learns
one cost over twelve future steps, while some evaluated tracks have only a short
labelled future prefix. In the retained easy-case errors, incomplete futures
account for about 77% of gross harm in deathCircle and 75% in gates. Many complete
training trajectories also benefit overall while being worse over their first
few steps. I have implemented and checked prefix-level supervision without using
future label availability as an input. Retraining and a fixed policy comparison
come next; this [diagnosis and target-interface repair](outputs/publication_readiness_2026_09/log_cost_v1/conditional_support_diagnosis.md)
does not yet improve a deployed model or justify excluding short tracks.

I have now fixed the next comparison before training: two equally sized risk
heads learn either twelve copies of the full-trajectory cost or twelve distinct
prefix costs. The same forecasts, fitting rows, seeds and training budget are
used in both arms. Equal-switch-count controls will test whether any protection
comes from better risk discrimination rather than simply switching less often.
The [registered experiment](outputs/publication_readiness_2026_09/prefix_cost_v1/registration.md)
has now completed all 24 fits. The result is negative: guarding every prefix
reduces ADE gain to 1.20%, versus 4.19% for the matched terminal-cost control;
it also loses at the same switch count. One scene/seed still exceeds the easy
ceiling. Almost every added veto triggers at the first predicted step, rejecting
many forecasts that would improve the full trajectory. The [results and failure
analysis](outputs/publication_readiness_2026_09/prefix_cost_v1/conclusions.md) keep
this tradeoff explicit. I have not deployed the repair. Independent calibration
and confirmation remain unresolved.

I have also tested changing the action rather than the risk threshold: introduce
the neural forecast gradually, keeping the first point at the causal baseline.
A uniform blend is matched to the same forecast displacement per query. All 24
new risk-head fits are complete. The gradual intervention preserves the positive-error
easy ceiling in every scene and seed, but gains 3.40% ADE versus 3.64% for uniform
blending. It also harms one case where CV was exact. Identical-choice controls show
that both the temporal shape and the learned choices cost useful accuracy; risk
is still underestimated on accepted cases. This is a documented protection/accuracy
tradeoff, not a new deployable winner. The [verified results and failure diagnosis](outputs/publication_readiness_2026_09/temporal_intervention_v1/conclusions.md)
remain separate from the pinned manuscript. Independent confirmation is still
missing; I have not replaced the primary ADE criterion with the better endpoint
result or changed the deployment.

## Research Question

My primary task is **eight observed annotation steps to twelve predicted steps**.
Raw-frame `t+50` is a separate supplement. I study when a neural forecast adds
value over a strong causal baseline, how to estimate the harm from switching,
and whether decisions for interacting agents should be made together.

I have adopted a transparent evaluation amendment: native-coordinate ADE/FDE
within each dataset, with relative ADE improvement averaged equally over fixed
physical scenes. This follows a diagnosed weighting problem, so it is post-hoc
protocol development, not a new independent test. I retain the old normalized
scores and every negative result. The change does not establish a model gain.
[Decision and fixed first readout](outputs/publication_readiness_2026_09/native_metric_v1/decision.md).

I have now completed that matched training comparison on the full admitted source
population: 24 real Torch fits, four excluded source sites and three seeds. With
the same model, batches and budget, native-loss training improves ADE by **7.63%**
over causal constant velocity, compared with **2.18%** for the old-loss control.
The direct improvement over that control is **5.56%**. All four source scenes and
all three seeds improve; the scene-bootstrap interval against CV is [5.96%, 9.30%].

This is a useful predictor result, but not yet safe intervention. The new model
also increases error on some paths that CV predicts exactly. Those zero-reference
errors cannot be hidden behind an undefined percentage. I keep this as a research
candidate, not a new deployment, and retain all old scores and failed experiments.
The four scenes have already been explored, so this is not independent confirmation.
[Controlled result, safety failure and reproduction](outputs/publication_readiness_2026_09/native_forecast_v1/conclusions.md).

I have now completed the training-lineage repair for the intervention head:
18 additional pair-excluded fits, 72,000 updates and twelve physically separated
cost-training views. Their upstream predictors exclude both the row's own scene
and the head's validation scene. All 5,484 fixed checkpoint-replay predictions
match exactly, and all cost entries pass a separate arithmetic check. The twelve
existing outer-held predictors are retained. This prepares honest cost-learning
data; it does not yet establish a safer selector or independent calibration.
[Completed training and limits](outputs/publication_readiness_2026_09/native_nested_v1/conclusions.md).

The first cost-head comparison is now complete: twelve ridge and twenty-four
small neural fits, using three seeds and clean nested training views. Penalizing
harm underestimation reduces harmful interventions, but also gives up most of
the forecast gain. With the fixed conservative rule, ordinary neural regression
improves ADE by 1.29%; the asymmetric loss improves it by 0.34%. Neither protects
every path that CV predicts exactly. The latter harms one such query in one
seed, so I do not promote it or select only the other two seeds.

This raised the next question: does the cost head rank safe opportunities better,
or does it merely switch less often? All 36 heads replay, independent arithmetic
agrees, and the negative safety result is retained. These are developmental
results on explored source scenes, not a new deployable model or independent
calibration. [Full comparison and failure analysis](outputs/publication_readiness_2026_09/native_gain_harm_v1/conclusions.md).

The matched-intervention comparison is now complete. At the same 0.84% switching
rate, asymmetric ratio ranking improves ADE by 0.34%, ordinary MSE ratio ranking
by 0.43%, and MSE net-gain ranking by 1.19%. The asymmetric loss harms fewer
exact-zero-CV outcomes, but sacrifices accuracy; every nontrivial control still
fails strict protection. This separates the value of ranking from simply doing
less. This motivated separating net-gain allocation from an explicit protected-risk
target instead of treating a benefit/harm ratio as a safety certificate. These are fixed
offline controls on explored sources, not online deployment or untouched tests.
[Matched counts, paired contrasts and limits](outputs/publication_readiness_2026_09/native_matched_coverage_v1/conclusions.md).

That protected-risk experiment is now complete: 24 matched neural heads and
72,000 updates. The new guard retains 5.57% source ADE improvement, but still
harms 25 zero-reference query/seed instances and degrades the positive-easy
diagnostic by 7.16%. A simple past-stop veto does slightly better. The broader
harm guard avoids these observed zero-reference harms by almost never switching;
matching its capacity leaves only eight decisions. I do not count abstention or
a high event AUROC as a successful safety mechanism. The next question is how to
learn reliable harm estimates specifically where intervention is proposed, with
clean calibration and enough event support. No new policy is deployed.
[Protected-risk results and reproducible negative evidence](outputs/publication_readiness_2026_09/native_protected_risk_v1/conclusions.md).

I then tested a more structured risk target. When CV is exactly correct, harm
from replacing it is simply the known distance between the two forecasts. A
48-fit feature/loss comparison uses that identity instead of asking a network
to learn every rare harm magnitude from scratch. Cost MSE improves in every
comparison with the previous direct head, but safe decision-making does not.
Extra history-consistency features and the geometric loss add no practical
selection gain. The useful control is simpler: the existing strict cost rule
plus a past-stop veto retains 1.29% ADE improvement without observed harm to
complete zero-reference queries across the three seeds. It still has unobserved
future outcomes and no independent calibration, so I keep it as a research
reference, not a new deployed model. Better risk regression alone is not the
contribution I need to establish.
[Geometric-risk results, simple control and limits](outputs/publication_readiness_2026_09/native_geometric_risk_v1/conclusions.md).

Before testing joint decisions, I checked that the agents can actually be placed
in the same scene. This caught 52 false identity links in a position-only assembly
probe and 3,036 ambiguous neighbor slots. I rebuilt the links from source IDs and
past observations. The resulting cache covers 175,756 forecast targets and keeps
145,805 additional context rows explicit rather than pretending every visible
agent has a neural prediction. Some context has too little history even for CV.
This fixes an experimental prerequisite, not the model's accuracy or safety.
It also confirms why observed protection is not a guarantee: the frozen control
still selects incomplete or absent future outcomes.
[Scene repair, coverage and limits](outputs/publication_readiness_2026_09/native_scene_context_v2/conclusions.md).

That fixed joint comparison is now complete. At matched intervention counts and
predicted-harm budgets, joint and unary-geometry decisions are identical across
all three seeds. Only 88 scene/seed queries have a non-additive opportunity;
exhaustive checking confirms that this is not a solver failure. The geometry
proxy gets smaller than independent selection, but forecasting does not improve.
I therefore keep joint selection as a negative control, not a claimed innovation.
The simpler conservative reference still gives 1.29% developmental ADE gain,
with unresolved missing outcomes and no independent safety calibration.
[Full result and reproducible diagnosis](outputs/publication_readiness_2026_09/native_joint_controls_v1/conclusions.md).

I next tested a smaller, explicit hypothesis: constrain predicted benefit and
harm by the known disagreement between the frozen forecasts. All 36 matched
cost-head fits are complete. The primary comparison improves source ADE gain by
1.47 percentage points, but it still harms perfectly CV-predictable paths and
fails easy preservation in one seed. At the same intervention count, its
advantage is only 0.047 points. The bound alone is not a safety mechanism.

A predeclared fraction-loss variant gives 2.44% ADE gain with no observed harm
on complete zero-CV paths and a 0.56% improvement on the positive-easy diagnostic.
This is a promising development tradeoff, not a new deployed model: incomplete
selected outcomes remain unknown, conditional harm is still underestimated, and
all four sites have informed model design. I am keeping the failed primary
protection result alongside that favorable secondary result.
[Matched cost heads, uncertainty and limits](outputs/publication_readiness_2026_09/bounded_cost_v1/conclusions.md).

I also completed a matched native-loss run of the public EqMotion core: four
source sites, three seeds, identical training rows, draws and update budgets.
It improves ADE by 11.04% over CV, compared with 7.63% for my local Transformer.
The paired difference is 3.41 percentage points, with a conditional site interval
of [0.94, 6.07]. I therefore cannot claim that my Transformer is the stronger
prediction architecture in this setting.

EqMotion also increases positive-easy error by 35.25% and harms many paths that
CV predicts exactly. That keeps the central research question open: a better
average forecast still needs reliable intervention control. The first readout
and complete checkpoint replay pass a separate arithmetic check; neither model
is newly deployed.
This fixed-head comparison is not reproduction of the author's best-of-20
benchmark, and the explored scenes are not independent confirmation.
[Strong comparator, failures and scope](outputs/publication_readiness_2026_09/native_eqmotion_v1/conclusions.md).

I then transferred the existing cost heads to EqMotion without retraining or
changing their thresholds. The fixed fraction-based rule retains 3.33% ADE
improvement, but easy error rises 3.78%, above the 2% limit. Its small advantage
over direct cost regression is not resolved by the scene interval. The heads
underestimate switching harm in every site/seed view, and candidate-rollout
features move outside the distribution on which those heads were trained.
This is a failed transfer, not a safe model. My next step is to build properly
cross-fitted EqMotion training predictions before learning its intervention
costs; changing a threshold on these outcomes would not answer that question.
[Frozen transfer, feature diagnosis and limits](outputs/publication_readiness_2026_09/cost_head_transfer_v1/conclusions.md).

The eighteen pair-excluded EqMotion fits needed for that follow-up have now
finished, taking 7.75 hours locally without reducing the registered budget.
Their prediction caches pass the row-level cost and source-exclusion audit,
with fixed-block checkpoint replay. The 36 predictor-specific cost-head fits are also complete,
with the same thresholds and a comparison at common intervention counts.
Refitting improves positive-easy error by 1.79% and retains 1.61% ADE gain over CV,
but it does not beat the old transferred rule's 3.33% gain. The registered primary
comparison therefore fails. On one scene it rejects high-benefit predictions
because it overestimates their harm; on its own selected rows it still
underestimates harm. This is a narrower protected development tradeoff, not a
solved risk model or a new deployment.
[Predictor-specific results and remaining failure](outputs/publication_readiness_2026_09/eqmotion_cost_refit_v1/conclusions.md).

I checked whether that failure begins only on an unfamiliar scene. It does not:
the fraction-based cost loss already understates benefit and overstates harm on
high-disagreement fitting examples. Native-error training ranks those examples
better but has failed protection elsewhere. I tested one fixed intermediate
loss weighting across all twelve scene/seed combinations, keeping forecasts,
sampling and thresholds unchanged. It repairs much of that tail-ranking error,
but loses many modest low-risk opportunities in another scene. Strict-policy
ADE gain falls from 1.61% to 1.10%, so the primary comparison fails again.
Matching intervention counts shows a small ranking gain over native loss, but
easy degradation is still 7.50% there. I do not promote that secondary result
or relax the threshold to make the experiment pass. The next question is whether
the remaining tradeoff reflects limited fitting capacity or incomplete
optimization. That fixed comparison is now complete. Increasing the training
budget has a larger effect than simply widening the head: the registered
wide/long policy improves ADE by **3.73%** over CV, versus **1.61%** for the
earlier protected fraction-loss control. The paired improvement is **2.12
percentage points**, with a conditional scene interval of **[0.52, 4.53]**.
All three seed aggregates preserve easy cases, and no complete zero-CV query
is harmed. This is a real development gain, not a new deployment.

There are important limits. Easy error still rises by **2.95% in deathCircle**,
selected harm is underestimated, and missing future labels prevent a complete
safety assessment. The old fraction control also had a smaller training budget,
so I could not attribute the whole gain to the loss function. I have now
completed the fair follow-up: 24 new native/fraction control fits with the
same wider head, longer budget and training samples. The intermediate objective
retains **3.73%** ADE gain, versus **3.07%** for native cost and **1.66%** for
fraction cost. Its advantage over fraction is supported by the conditional
scene interval, but its **0.66-point** advantage over native has an interval
of **[-0.27, 2.19]**. The registered claim required both, so it has not passed.

This result removes the budget mismatch without hiding the remaining failure.
All 36 checkpoints replay exactly, and separate arithmetic confirms the scores.
Selected harm is still underestimated under every objective; the deathCircle
easy failure remains. I keep the candidate as developmental evidence, not a
new deployment or proof that the proposed loss is generally superior.
Independent calibration and final confirmation are still missing.

I have also traced the remaining harm error on the same samples, not just each
model's different selections. Global fitting estimates are usually conservative,
but the model becomes optimistic on the rows it chooses to replace. This is
visible even during fitting and gets worse on an excluded scene. I have now
completed the fixed follow-up: twelve new heads put more fitting weight on that
decision region, with the same model, budget and inference rule. ADE improvement
rises from **3.73% to 4.10%**; the paired gain difference is **0.37 points**,
with a conditional scene interval of **[0.17, 0.60]**.

The accuracy gain is real within this development comparison, but the protection
repair fails. Easy error still rises **2.93% in deathCircle** and **2.15% in gates**
on seed average. The model fits the old selection region better but remains
optimistic on its own new selections. At equal intervention counts, its accuracy
is slightly worse than the previous head. I therefore do not deploy it or claim
that weighting alone solves risk estimation. Independent calibration and final
confirmation remain open requirements.

I have now checked what an honest calibration split would require. Removing one
site from the cost-head rows is insufficient: the predictors that generated the
remaining targets still learned from that site. All 36 proposed inner reuse cases
fail this check. A new refusal guard catches this before calibration reads. The
existing outer-held fitting exclusion remains valid; these are different claims.
A fully nested repair needs twelve new predictor fits and thirty-six cost heads,
yet still offers one calibration site per policy. Independent scene support is
the priority; a source rotation cannot be relabeled as independent safety evidence.
[Calibration feasibility and remaining decisions](outputs/publication_readiness_2026_09/calibration_support_v1/conclusions.md).

I am now checking independent data support before claiming that this protection
transfers. A fresh audit of all 30 local TRAF annotation files finds unresolved
box conventions, class identities and camera/site grouping; none is admitted
to a new experiment. I also located DroneCrowd's separate annotation archive
and its academic-use terms. Its release README explicitly says validation is
sampled from test, so I will not treat those folders as independent calibration
and confirmation. I have now pinned the five small official metadata files and
checked all 112 clip IDs. The conversion code also shifts frame/agent indices
and removes visibility information, so a derived MAT cannot stand in for a
verified causal history. Those checks are implemented; original XML, camera
motion and independent physical sites still need review. No new external score
is reported from this intake work.
[Metadata evidence, tested checks and remaining limits](outputs/publication_readiness_2026_09/dronecrowd_metadata_v1/conclusions.md).
[Source audit and acquisition status](outputs/publication_readiness_2026_09/traf_intake_v1/conclusions.md).

I also checked the annotation tool named in the DroneCrowd paper. One pinned
VATIC exporter interpolates tracks before writing XML without the generated
flag. This does not establish what happened in DroneCrowd's actual release, but
it means an original XML is not automatically evidence of sensor-time causality.
I keep offline annotated-position forecasting distinct from that stronger claim;
the next intake check must trace the actual annotation producer, not just frames.
[Source evidence and a tested dependency counterexample](outputs/publication_readiness_2026_09/annotation_export_provenance_v1/conclusions.md).

[Conditional diagnosis](outputs/publication_readiness_2026_09/cost_budget_matched_v1/conditional_diagnosis.md).
[Registered fitting repair](outputs/publication_readiness_2026_09/conditional_cost_v1/registration.md).
[Completed repair, negative safety result and verification](outputs/publication_readiness_2026_09/conditional_cost_v1/conclusions.md).
[Training-versus-transfer diagnosis](outputs/publication_readiness_2026_09/eqmotion_cost_fit_forensics_v1/conclusions.md).
[Fixed intermediate-loss result](outputs/publication_readiness_2026_09/tempered_cost_v1/conclusions.md).
[Fixed capacity/duration design](outputs/publication_readiness_2026_09/cost_capacity_v1/registration.md).
[Completed factorial, positive primary result and remaining failures](outputs/publication_readiness_2026_09/cost_capacity_v1/conclusions.md).
[Equal-budget objective controls](outputs/publication_readiness_2026_09/cost_budget_matched_v1/registration.md).
[Completed fair comparison and its limits](outputs/publication_readiness_2026_09/cost_budget_matched_v1/conclusions.md).

I am prioritizing that focused accuracy-versus-harm question over expanding the
model's scope. If the reference predicts a group exactly, I report absolute harm
and do not manufacture a percentage by adding a denominator. I retain strict
protection there rather than introduce a convenient pixel allowance. This is an
empirical research criterion, not a guarantee under unseen distribution shift.
[Research choice and its limits](outputs/publication_readiness_2026_09/native_nested_v1/research_choice.md).

In the preceding cache-only readout, across four already explored SDD source
sites, causal constant velocity remained the strongest fixed control. A
future-informed per-query oracle has 28.63% ADE headroom, but the old neural
predictions still lose on their original static-history subset (three-seed mean
-5.38%). The oracle is not a model result, and that subset is not full-population
neural coverage. This gives me a clearer next experiment without hiding the
failed one. [Paired results and limitations](outputs/publication_readiness_2026_09/native_metric_v1/conclusions.md).

The current paper direction is reliable baseline-relative intervention with
support-aware fallback. Scene-level coupling remains a tested negative control,
not an established contribution. A Transformer, JEPA encoder, cost head or
triangle-inequality bound is not novel just because it is part of this system.
Each component has to earn its place through matched comparisons and useful
independent results. M3W remains the longer-term project, not a reason to make a
broader claim than these experiments support.

I also distinguish training windows from genuinely different situations. A recent
[event-support audit](outputs/publication_readiness_2026_09/source_event_support_v1/conclusions.md)
maps 15,430 stationary-history windows to 1,457 annotation episodes. The 207
larger-excursion windows come from only 47 scoped tracks. Nearly all already have
moving neighbors, so missing neighbor slots do not explain that subset. The
completed episode-balanced training comparison makes prediction substantially
worse. It also reveals an important distinction: changing which windows are
sampled changes the expected training objective, even with the same per-row loss.

## Current Evidence

I have also tightened the test of the proposed interaction mechanism. A joint
policy can beat a simple selector just because it adds better single-agent
geometry penalties, even when no true coupling is present. The new matched
control retains those penalties and removes only the pairwise coupling. Its
implementation passes exhaustive and real past-input checks, including a
repaired numerical solver failure. This makes the comparison more informative;
it does not establish a new forecasting gain.
[Mechanism control and numerical evidence](outputs/publication_readiness_2026_09/interaction_controls_v1/conclusions.md).

I have now completed that comparison for all 24 fixed Transformer/EqMotion
seed, cost-head and policy combinations, without retraining or changing the
evaluation rules. Joint versus geometry-aware independent selection gives
21 identical, two slightly better and one slightly worse ADE results. None of
the new controls preserves easy cases within 2%. The evidence does not support
joint selection as an effective main contribution yet. These are already
explored development recordings from one physical site, not an independent
generalization test.
[Complete comparison, including negative results](outputs/publication_readiness_2026_09/frozen_interaction_v1/conclusions.md).

I then traced why all of those controls passed their predicted harm budget but
failed easy preservation. Two problems remain: the cost heads often underpredict
observed harm, and a small average absolute harm over the whole scene does not
protect a small relative error on easy agents. In one fixed comparison, observed
labels already prove that 529 of 970 queries exceed the realized budget. In
another, queries that really are within budget still contribute 57.88% of easy
harm. Missing selected outcomes remain unknown, not zero. This diagnosis keeps
all 72 comparisons and changes no model or threshold; a replacement risk target
still needs a registered experiment and independent calibration data.
[Risk forensics and its limits](outputs/publication_readiness_2026_09/frozen_risk_forensics_v1/conclusions.md).

The follow-up now locates that problem before cross-site transfer. I replayed
all twelve frozen cost heads on their own fitting rows. Every head beats a
constant on overall harm MSE, yet 23 of 24 fixed eligibility groups have negative
realized mean gain despite predicting positive gain. Source-batch replay finds
no ordering or scale mismatch in the checked samples. The current readout fits
global costs better than it identifies reliable interventions; simply training
longer or rescaling the overall mean is not an evidence-backed fix.
[Fit diagnosis, including the favorable exception](outputs/publication_readiness_2026_09/cost_head_fit_forensics_v1/conclusions.md).

I have checked that diagnosis against work on decision-focused learning and
conditional calibration. Switching to a ranking loss is not, by itself, a new
method. The unresolved question is whether costs are reliable for the actual
scene-level intervention and its easy-case constraint. Four executable
mathematical examples clarify why global fit, score calibration and a pooled
risk budget cannot substitute for those checks. They are synthetic explanations,
not new forecasting gains; the subsequent evaluation amendment is documented above.
[Prior work, derivations and tested examples](outputs/publication_readiness_2026_09/conditional_decision_v1/prior_work_and_method_boundary.md).

Before the next cost-head experiment, I checked whether the old OOF caches could
supply a genuinely held-out validation fold. They cannot simply be split again:
the predictors behind the remaining training rows have already seen that fold.
All 18 reuse attempts fail this recursive check, even though the original OOF
forecasts themselves are valid. I added a pre-fit check that rejects this
shortcut and identifies which outer-held predictors remain reusable. The next
head comparison needs nested producer exclusion, not just new selector weights.
This is a validation-design finding, not a forecasting gain.
[Verified reuse boundaries and the concrete repair](outputs/publication_readiness_2026_09/cost_validation_lineage_v1/conclusions.md).

I also repaired a reproducibility gap: a completed ridge run could report a
verified resume even after its OOF cache changed. The versioned entrypoint now
checks the entire completion dependency chain. All six real frozen ridge heads
match their earlier snapshots; the defect was reproduced and blocked using
temporary synthetic training. Old weights, source hashes and scores stay intact.
[Recovery behavior, verified assets and remaining limits](outputs/publication_readiness_2026_09/cost_completion_v2/repair_and_verification.md).

The latest broader source audit changes my diagnosis of the current task. Across
175,756 past-indexed queries, the old per-query normalization makes 6,864
static-start windows account for 99.75% of complete-label CV error. The same
windows account for only 0.67% in annotation pixels. A numerical scale floor is
therefore making the overall score almost entirely a static-start test.
Moving-history baseline-oracle headroom is 15.22%, but it falls to 0.038% in the
full normalized aggregate. This is an evaluation-weighting issue, not a new model
success. I retain the old metric and results alongside the adopted evaluation
amendment. That audit itself did not train a model or authorize deployment.
[Audit, raw checks and implications](outputs/publication_readiness_2026_09/source_population_v1/conclusions.md).

The preceding source experiment asks whether image downsampling hides useful motion.
I recovered all 25,300 past crops at native resolution, verified their exact
alignment with the old inputs, and fitted 64 fixed probability probes across
resolution and motion-window controls. Higher resolution improves measurement
support, but does not make this readout predict larger future changes reliably.
Training AUROC is about 0.79-0.80; the held-site average is about 0.48-0.49.
I retain the small favorable ranking contrasts alongside the worse probability
errors rather than treating them as a deployment result.
[Full comparison](outputs/publication_readiness_2026_09/source_motion_resolution_v1/conclusions.md).

The implementation runs, but the clean development experiments have **not yet
established a deployable neural advantage or a submission-ready method**.

| Question | What the completed evidence shows |
| --- | --- |
| Does the aggregate score represent ordinary motion well? | Not under the old normalization: 4.77% of complete windows contribute 99.75% of CV error. The new native-coordinate amendment is explicit; it does not turn old results into independent evidence or model success. |
| Do neural trajectory models beat strong motion baselines? | The fixed three-seed Transformer and K=1 EqMotion comparisons did not produce safe positive gains on the primary task. |
| Does longer training help? | Learning-rate decay produces a small source-training gain, but it does not transfer to the excluded source scene. |
| Does the tested RGB representation help? | The matched source comparison is negative. More input modalities are not automatically more predictive information. |
| Does cost-aware fallback help? | It reduces neural harm, but the fixed source readout still loses 1.246% to stationary CV. The unprotected control loses 1.744%. |
| Does satisfying a predicted global harm budget protect easy agents? | No. Frozen-risk forensics finds both observed cost underprediction and a mismatch between global absolute harm and conditional relative easy degradation. All 72 controls still fail the easy requirement. |
| Do scene-excluded candidate forecasts remain useful? | Twelve fresh fits all lose on their excluded site; equal-site gain is -5.016%. Fixed candidate/CV oracle headroom is below 0.53%, so another gate alone is not the next repair. |
| Does removing the static-target loss repair them? | No. Twelve matched new fits increase oracle headroom to 3.760%, but actual gain is -98.719% and static-target harm is much larger. |
| Do raw annotation checks and past-box features explain the failure? | Small changes are common, but >10px queries contribute 53.25% of baseline error and still lose. Forty-eight fixed probability probes find no stable added-box benefit. |
| Do pretrained image features repair source transfer? | No. Thirty-six matched trajectory heads complete 360,000 updates. Geometry/current-image/eight-frame gains are -0.070%/-1.908%/-6.102%; all held fits are negative. |
| Does removing shared appearance repair the temporal model? | It reduces harm, but does not beat the baseline. Twenty-four fresh heads give -0.762% for centered input and -1.756% with RMS normalization; all held fits remain negative. |
| Does balancing exposure across annotation episodes help? | No. Twenty-four fresh heads complete 240,000 updates, but geometry and centered-image gains fall to -37.327% and -54.992%. The sampler changes the effective training objective and greatly increases static-target harm. |
| Does exact importance correction fix that objective shift? | It removes most of the added harm, but not the prediction gap. Another 24 heads/240,000 updates give -0.032% for geometry and -0.275% for centered images; all held fits remain negative. |
| Is gradient clipping sending training in the wrong direction? | The fixed-checkpoint training audit does not support a large direction reversal. Train-scale output conditioning removes logged clipping and most jitter, but 24 new heads still lose to CV: -0.000251%/-0.001342%. |
| Does explicit observed image motion repair the remaining gap? | No. Another 24 heads complete 240,000 updates; quality-only/motion gains are -0.000535%/-0.000840%. Sixteen probability probes show a weak larger-excursion ranking gain but worse probability error. No new deployment. |
| Does native resolution or a smaller motion window help? | Not with this fixed regional readout. All 64 probability probes complete; larger-excursion Brier worsens when motion is added in all four measurement variants. None beats the training-prevalence reference on that label at any held site. |
| Are the historical external selector gains independently verified? | No. Recording duplication, teacher exposure and test-based selection make those scores exploratory. |
| Is scene-level joint intervention validated? | Exact-count and geometry-aware independent controls now isolate the proposed coupling more carefully. Engineering checks pass, but predictive advantage and independent risk calibration remain unproved. |

The latest [fixed deferral readout](outputs/publication_readiness_2026_09/source_deferral_transfer_v1/conclusions.md)
retains all six trained endpoints and three matched controls. All three
cost-supervised seeds lose to CV; the conditional recording interval is
[-4.430%, -0.567%]. It concerns seven recordings of **one previously explored
site**, not independent confirmation. Exact replay verifies reproducibility,
not forecasting quality. Complete rejection returns the baseline and is not
a new prediction success.

The latest [candidate cross-fit experiment](outputs/publication_readiness_2026_09/source_crossfit_v1/conclusions.md)
completed all 120,000 updates across four internal site folds and three seeds.
Equal-site gain is -5.016%, conditional interval [-8.397%, -2.488%]. Most excess
error comes from predicted movement on stationary targets, but the remaining
moving-target predictions also lose on average. The experiment isolates producer
exposure, not the causal reason for the transfer gap. Bookstore and the main
evaluation remain unscored; no new model is deployed.

The [matched loss intervention](outputs/publication_readiness_2026_09/source_motion_candidate_v1/conclusions.md)
has now completed another 120,000 updates. Removing static-target gradients
makes the forecast less conservative, but it also worsens moving-target error.
Rotating its predictions retains most of the oracle headroom, so that headroom
alone is not evidence of accurate motion direction or usable neural dynamics.
All twelve models replay exactly; the scientific result is still negative.

The completed [motion-quality diagnostic](outputs/publication_readiness_2026_09/source_motion_quality_v1/conclusions.md)
aligns all 15,430 source queries to raw annotations. It distinguishes tiny
coordinate changes from larger excursions without deleting either group.
Past-box features do not repair cross-site motion probabilities. Interpolation
controls after the query also occur in 15,316 histories, reinforcing the
offline-annotation limitation rather than establishing real-time perception.

The completed [pretrained temporal comparison](outputs/publication_readiness_2026_09/source_pretrained_temporal_v1/conclusions.md)
adds frozen visual features without changing the cohort, loss or sampling budget.
Appearance improves training fit slightly but worsens excluded-scene prediction.
Eight-frame appearance loses another 4.194 percentage points relative to current
appearance. All 36 heads replay exactly; this confirms the negative result, not
a deployable visual dynamics contribution.

I then checked whether the temporal model was mostly fitting shared appearance.
The input audit found correctly aligned, non-identical historical frames, but
little within-window variation in the frozen features. The
[registered centering comparison](outputs/publication_readiness_2026_09/source_temporal_centered_v1/conclusions.md)
completed all 24 heads and 240,000 updates. Centering reduces the sequence model's
excess forecast error over CV from 6.102% to 0.762%; normalizing the variation
still increases error by 1.756% over CV.
Both remain worse than geometry alone. These are useful negative controls, not a
new deployable model. The remaining question is whether the observed histories
provide enough transferable information about independent state-change events.

The [episode-exposure experiment](outputs/publication_readiness_2026_09/source_episode_sampler_v1/conclusions.md)
keeps those inputs and all evaluation rows, but samples annotation episodes
equally during training. All 24 new held-site fits are negative. A
[post-hoc diagnosis](outputs/publication_readiness_2026_09/source_episode_sampler_v1/failure_analysis.md)
shows why this is not simply a training-runtime problem: every head improves
its reweighted training risk while worsening the original unweighted risk.
The sampled proportion of future-changing labels rises from 38.62-47.49% to
61.71-73.54%. Future labels are used only to describe this shift, never to build
the sampling groups or inference inputs. Exact replay confirms the failure;
it does not rescue the model.

I then ran a [matched importance-correction experiment](outputs/publication_readiness_2026_09/source_importance_sampling_v1/conclusions.md)
with the same episode draws and a loss weight that restores the original
expected risk. Geometry and centered-image excess errors fall to 0.032% and
0.275% over CV. This identifies and repairs the large sampling-induced harm,
but it does not create a useful neural candidate: all 24 new held fits still
lose, and adding these visual features still hurts. The distinction between
repairing training and demonstrating a prediction contribution matters here.

## Evidence and Reproduction

The detailed record is kept separately so that the project overview remains
readable:

- [Results ledger](README_RESULTS.md): complete experiment outcomes, failures and current evidence boundaries.
- [September research history](README_RESEARCH_HISTORY_2026_09.md): the detailed routes and diagnoses behind this summary.
- [Recording and teacher-lineage audit](outputs/publication_readiness_2026_09/recording_lineage_audit.md): why historical external gains cannot be treated as independent evidence.
- [Working paper](outputs/publication_readiness_2026_09/paper_working_draft.md): the research question, method proposal, results and missing evidence, not a finished submission.
- [Latest experiment reproduction](outputs/publication_readiness_2026_09/source_importance_sampling_v1/reproducibility.md): commands, hashes, replay checks and limitations.
- [Data-role contract](outputs/publication_readiness_2026_09/experiment_contract/implementation_and_limits.md): training, selection, calibration and confirmation boundaries.

The current observation contract uses supplied historical annotations. Some
annotations may have been interpolated using later controls; past-indexed
access therefore does not prove strict sensor-as-of availability. Future
targets are kept out of inference features, and previously explored scenes
cannot become independent tests by renaming their roles.

## What The System Looks At

The current M3W pipeline works with dataset-local top-down trajectories. It uses information that would be available at inference time:

- recent agent history;
- speed, acceleration, heading, curvature, and stop/go behavior;
- neighbor density and interaction signals;
- train-only scene or goal context when that context is legally available;
- causal baseline rollouts;
- dataset, scene, horizon, and domain metadata;
- risk heads for failure, gain, harm, and fallback decisions.

I also maintain a neural track with Transformer dynamics, JEPA-style representation learning, hybrid heads, waypoint prediction, and protected residual policies. Guarded selection, causal history windows, full-waypoint structure, domain-aware routing, and safety floors are the most promising routes in the historical experiments. Their external gains remain exploratory until the clean evaluation is complete; neither the selector nor the neural branch has earned a new deployment claim from this audit.

## What This Repo Is For

This repository is a research record. The most important rule in the project is that a result has to survive the boring checks: no future leakage, no test endpoint goals, no central-velocity shortcuts, no easy-case damage hidden inside aggregate gains, and no metric claims without calibration.

For a quick orientation:

| File or directory | What to read it for |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | Detailed results ledger and current evidence boundaries. |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | Chinese long-form summary of routes tried, failures, causes, and successes. |
| [`research_state.json`](research_state.json) | Machine-readable snapshot of the current project state. |
| `outputs/m3w_neural_v1/` | Neural world-model reports and model-card style summaries. |
| `outputs/stage42_long_research/` | Cross-domain safety, replay, full-waypoint, and paper-claim evidence. |
| `outputs/stage43_latent_state/` | Latent-state, graph/history/context, and reviewer-style validation reports. |

Large datasets, caches, checkpoints, videos, images, third-party data, and local virtual environments are intentionally kept out of git.

## What I Am Not Claiming

M3W is not a true 3D world model yet. It is not a foundation world model. SDD results are pixel-space unless calibration is verified. External results are dataset-local unless their geometry is verified. `t+50` and `t+100` are raw annotation-frame horizons, not seconds. Self-audited or inferred labels are not human gold labels.

Stage5C latent generative execution has not been enabled. SMC has not been enabled.

The current claim is narrower: this repo contains a protected 2.5D multi-agent world-state research system and an active neural dynamics track. Its historical external evaluation has identified independence failures that I am repairing before making new generalization or deployment claims. This external audit does not establish the status of every SDD experiment.

## Running Locally

On Apple Silicon, training should use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Focused checks for the new data and evaluation path:

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_deferral_development.py -q
.venv-pytorch/bin/python -m pytest tests/test_m3w_cost_sensitive_deferral.py tests/test_m3w_matched_coverage.py tests/test_m3w_citr_recordings.py tests/test_m3w_confirmation_evaluation.py tests/test_m3w_risk_calibration.py tests/test_m3w_eqmotion_adapter.py tests/test_m3w_development_evaluation.py tests/test_m3w_supervised_intervention.py tests/test_m3w_external_source_audit.py tests/test_m3w_experiment_contract.py tests/test_m3w_joint_intervention.py tests/test_m3w_causal_recordings.py tests/test_m3w_recording_lineage.py tests/test_stage44_worldcore.py
```

The legacy full suite (`python -m pytest tests`) includes integration training and can rewrite reports in the working directory. It is not yet an isolated, read-only smoke test; preserve existing experiment outputs before running it. The [local/CREATE runbook](outputs/publication_readiness_2026_09/local_create_runbook_zh.md) records the verified environment and recovery checks.

Training scripts are written around checkpointing, heartbeat logs, resume support, CPU/MPS-safe execution, and single-process dataloading.

## Next Step

Improve candidate utility before fitting another risk head. Exact importance
correction and output conditioning are now tested. They repair objective shift
and reduce numerical jitter, but the models still have almost no useful
candidate/CV oracle headroom. I will not turn that into another threshold sweep.
The [completed comparison](outputs/publication_readiness_2026_09/source_conditioned_readout_v1/conclusions.md)
makes the distinction clear: better optimization does not necessarily produce
better dynamics. The next question is whether raw past visual motion contains
predictive cues that frozen image pooling loses, after accounting for crop
movement and occlusion. That input investigation has not yet run. More weight
on rare windows cannot create independent events or missing cues.

The loss, annotation, visual-feature, temporal-centering and episode-sampling
controls remain available, including their negative results. No policy or test
threshold has been changed to rescue them. OOF labels also do not automatically
permit a second-level validation split: every upstream producer must exclude
the risk head's validation scene.
[Provenance boundary](outputs/publication_readiness_2026_09/source_crossfit_v1/method_and_limits.md).

The larger goal is unchanged: demonstrate useful neural dynamics, compare
independent and joint intervention at matched coverage, preserve easy cases,
and obtain genuinely independent calibration and confirmation. More overlapping
windows cannot substitute for more independent scenes. I am working toward
CVPR 2027, not claiming that implementation progress guarantees a publishable
result or acceptance.

When a route fails, I keep the evidence. A successful method must show where
it improves the baseline, where it does not, and how the result can be reproduced.
