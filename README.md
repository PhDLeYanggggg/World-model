# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling.

The question behind the project is simple:

> If I can see a scene, the agents in it, their recent motion, and their local interactions, can I predict what happens next more reliably than strong causal motion baselines?

I started this repo to answer that question carefully, not just to collect a nice-looking demo. The work here includes the models that improved results, the ones that failed, the leakage checks, the safety rules, and the notes that keep me honest about what the evidence does and does not prove.

## Read the Current Study

I am checking whether the tree comparator's protection comes from better risk
ranking or simply fewer switches. The [fixed-count control](outputs/publication_readiness_2026_09/risk_ranking_v1/registration.md)
keeps the forecasts and switch counts unchanged and compares relative harm with
net-gain ranking. Its readout is pending; it is not another training run or a
new deployment.

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
