# Learning When Neural Motion Forecasts Help: Cost Prediction, Selected Harm, and Temporal Supervision

English development-evidence manuscript, 5 October 2026. This revision brings
together two separate studies; it does not pool their errors, observations or
uncertainty. The [earlier SDD manuscript](../evidence_manuscript_v2/manuscript.md)
and its original results remain unchanged. This is not a submission-ready paper.

## Abstract

A neural trajectory predictor can improve average accuracy while degrading
predictions already handled well by a strong causal baseline. We investigate
whether learning baseline-relative benefit and harm can identify useful neural
interventions without concealing damage to easy cases. Two development studies
expose a gap between accurate cost prediction and reliable policy decisions.
On four explored Stanford Drone Dataset sites, a tree cost estimator outperforms
neural risk ranking at matched intervention counts, while the original unmatched
superiority test fails. On twelve explored EuropeanSquares localities, improved
training fit is followed by worse recording-held cost prediction. Restricting
past-quality extrapolation reduces normalized validation score error, but leaves
selected-risk violations unresolved. A subsequent analytic probe finds temporal
error structure beyond leaf-local whole-trajectory means: the signed step-error
MSE difference is -0.094342, with a nominal locality-bootstrap interval of
[-0.158824, -0.038614]. The same comparison on the original selected cohort is
inconclusive. We therefore register, but do not yet report results for, matched
temporal-auxiliary neural cost-head training. These findings motivate testing
selected utility and harm separately from prediction loss. They do not establish
calibrated safety, independent cross-scene confirmation or world-model success.
All forecasting is in image-local coordinates and raw annotation steps.

## 1. Introduction

Neural motion forecasting is often evaluated by its average displacement error.
For an application that already has a useful motion baseline, this is not the
only relevant question. A replacement may reduce error on difficult trajectories
and introduce avoidable error on simpler ones. Evaluating only the retained
neural predictions can also obscure the errors incurred by the complete system.
We ask when a neural prediction is worth using relative to a specified fallback,
and what evidence is needed before calling that decision reliable.

M3W is a broader research programme on multimodal multi-agent world models.
The present paper studies a narrower component: supervised decisions between
frozen forecasts. It does not demonstrate a new video world model, generative
rollout or end-to-end multi-agent dynamics contribution. This distinction matters
because improving a selector or a cost head does not, by itself, improve the
dynamics learned by its upstream predictor.

The working hypothesis is that useful intervention requires three separately
testable properties: accurate conditional benefit and harm estimates, a decision
rule that preserves those properties on its selected population, and independent
evidence that this relationship survives new scenes. Scene-level composition is
an additional hypothesis when independently chosen forecasts conflict. None of
these properties follows from low overall training loss.

The completed studies support a diagnostic account, not the complete hypothesis.
We preserve unsuccessful primary comparisons, conventional strong controls,
unknown outcomes, unsupported risk estimates and negative selected-cohort
results. The main empirical lesson is specific: in the examined development
populations, improved global cost fit repeatedly fails to establish safer
intervention. The next registered experiment asks whether temporally resolved
supervision changes that outcome under an otherwise matched neural cost model.

## 2. Related Work and Contribution Boundary

Motion forecasting supplies the candidate trajectories. EqMotion models
equivariant motion and invariant interactions [1]. Our earlier SDD study uses
an adaptation of its author implementation with a fixed forecast and the local
coordinate/loss protocol. This is not a reproduction of its published
best-of-many benchmark. JFP explicitly models dependencies between future agent
predictions [2]; joint prediction and pairwise compatibility are not new ideas
introduced here.

Regression with multi-expert deferral treats both joint predictor/router
learning and routing around pretrained predictors, with regression costs [3].
Our two-forecast decision problem overlaps with this established setting.
Predicting costs rather than an oracle class and returning a fallback forecast
are not sufficient novelty claims. Any claimed contribution must instead be
supported by the specific loss, support treatment or scene-level decision
mechanism, with matched evidence against existing alternatives.

Learn then Test formulates risk calibration using hypothesis testing [4].
Conformal Risk Control provides an expected-risk result under conditions that
include exchangeability and an appropriate bounded monotone loss family [5].
A learned harm threshold is not either procedure. Overlapping windows do not
become independent calibration examples, and changing selected agents can make
the realized harm ratio nonmonotone.

Conformal Policy Control is a closer reference for regulating an optimized
policy relative to a reference policy [6]. Its likelihood-ratio construction
requires policy probabilities and appropriate calibration conditions. Our
deterministic forecast selector has not implemented that construction or
established its assumptions. The comparison rules out claiming generic
reference-policy protection as our invention; it does not supply a guarantee
for the current selected-harm ratios.

The unresolved contribution is therefore deliberately narrow: whether temporal
cost supervision and selection-aware evaluation provide reliable additional
utility over strong cost estimators in this forecasting task. The present
results motivate this question but do not answer it positively. The literature
review is targeted, not an exhaustive novelty certification.

## 3. Task and Evidence Roles

### 3.1 Two separate development cohorts

Both studies observe eight annotation steps and predict twelve, using raw
stride twelve. Twelve predicted steps are not raw-frame t+50. Raw-frame t+50
belongs to a separate supplemental task, and no result is converted to seconds
or metres here.

The earlier SDD study [7] contains 175,756 past-eligible pedestrian queries from
33 recordings in four explored physical sites: coupa, deathCircle, gates and
hyang. Observed-point ADE is available for 172,957 queries, final-step FDE for
144,010, and complete future positions for 143,918. These are overlapping
windows, not independent people or independent trials. Historical annotations
may incorporate interpolation using later annotation controls. Past-indexed
access therefore supports an offline annotated-history contract, not a proven
real-time perception pipeline. The original
[data and lineage discussion](../evidence_manuscript_v2/manuscript.md#3-task-data-and-evidence-roles)
is retained.

The later study uses released EuropeanSquares detector tracks [8], not human-gold
trajectories. Twelve already-explored localities provide the source-development
population. The source index records 318,969 unique entries before repeated
use across rotations; the current frozen readout contains 596,988 row
occurrences across 72 source/head combinations, including 14,076 wholly unknown
outcomes. These counts describe different levels of reuse and must not be added
together. The three head seeds are 17, 29 and 43, not three new forecaster
trainings. Current cost studies use whole-recording TRAIN/validation partitions
within these explored localities. Recording-held validation is not independent
locality-held confirmation. The
[asset card](../european_fixed_producer_roles_v1/data_and_asset_card.md) and experiment-specific
protocols document the earlier producer role rotations and later frozen reuse.

SDD displacement gains and European normalized cost-error contrasts are reported
separately. No cross-dataset average is constructed. Historical Stage26/37
figures have different horizons, populations and development exposure; they are
not inserted into this paper as comparable independent-test baselines.

### 3.2 Causal input and producer exclusions

Inference inputs comprise past-indexed target/neighbor histories, past-derived
motion and quality descriptors, and causal baseline/candidate rollout
diagnostics. The European cost interface has 380 features. No future endpoint,
future remaining track length, central velocity or test-endpoint goal is added
to the inference interface. Future positions and their observation masks serve
only fitting losses and offline evaluation. Missing future labels cannot
determine deployment eligibility.

Producer exclusions, preprocessing sources and whole-recording partitions are
part of the frozen lineage, not reconstructed by relabelling a file as test.
All twelve current European localities have informed development. Reserved
selection, calibration and confirmation roles remain unopened in these studies.
This manuscript assembly reads public aggregate files only; it does not perform
a fresh trajectory-level leakage audit or reopen those roles.

## 4. Baseline-Relative Cost Learning

### 4.1 Benefit, harm and the easy event

Let r(x) denote the frozen reference forecast and n(x) the candidate forecast.
For a labelled example, let R be the reference displacement loss and N the
candidate loss, measured on the same observed future support. Define

    B = max(R - N, 0),        H = max(N - R, 0).

The European head predicts five conditional moments: B, H, R, ER and EH,
where ER = eR and EH = eH. The event e uses the original TRAIN-derived easy
cutoff on positive constant-velocity error. Importantly, that CV-defined event
and the reference forecast r are not interchangeable: the reference can be a
TRAIN-selected protected floor. True e is a supervision/evaluation label, not
an inference-time feature.

The bounded moment decoder respects nonnegativity, ER <= R, EH <= H and the
causal forecast-disagreement envelope. Decision scores are predicted net gain,
predicted all-risk excess and predicted easy-risk excess:

    B_hat - H_hat,   H_hat - 0.02 R_hat,   EH_hat - 0.02 ER_hat.

Selection additionally obeys the frozen causal eligibility/support guards.
Unsupported cases return the reference forecast. Satisfying predicted scores
does not imply that observed or population risk is controlled. Selected easy
risk is a ratio of weighted positive harm to weighted reference cost, not the
probability of failure. Zero or missing denominators stay undefined, not safe.

### 4.2 Controlled estimator changes

The European comparison retains the original forest, an additive quality
adjustment, a positive-harm Poisson control and a squared-cost head. The cost
support diagnostic changes no fitted models. It reconstructs training and
validation prediction errors and localizes their discrepancy by outcome moment,
selection status and TRAIN feature support.

The subsequent TRAIN-identity extension clips seven past-quality coordinates
to the known-TRAIN range inside each fixed leaf. Leaf routing and every known
TRAIN prediction remain unchanged. This tests a specific extrapolation repair,
not a new risk guarantee. It cannot be credited with policy improvement merely
because its global validation MSE decreases.

### 4.3 Temporally resolved targets

For each observed future step, define d(t) as candidate error minus reference
error. The analytic temporal probe estimates d(t) and reference error(t) within
frozen causal leaves using TRAIN-only, query-balanced means. A leaf-local
whole-trajectory mean repeated over time and a global per-step mean are retained
as distinct controls. Missing steps never enter target means; unsupported
leaf-step cells use the TRAIN global mean and are counted explicitly.

Temporal supervision must not silently redefine the primary harm target. With
the same observed-step weights, let G_H = mean(max(d(t), 0)) and
G_B = mean(max(-d(t), 0)). Then

    H = max(mean(d(t)), 0) = G_H - min(G_H, G_B).

This is an accounting identity, not a new theoretical contribution. Replacing H
by G_H would change the scientific risk definition. The completed diagnostic
leaves original actions, primary moments and the two-percent budget untouched.

### 4.4 Unknown outcomes and matched policies

Errors cannot be imputed as zero when trajectories disappear. Where neither
future error is observed, a causal disagreement envelope bounds the unknown
benefit-minus-harm term. Known partial-label measurements remain distinct from
complete-future accuracy. Completeness is reported rather than used to remove
inconvenient policy decisions.

Policy comparisons must also account for selection volume. The registered new
readout uses a common original-selected cohort and same-query minimum-count
comparisons in addition to each policy's own selection. It distinguishes a
difference of two lower utility bounds from a valid lower bound on their paired
difference. Shared unknown selections cancel in the latter; exchanged unknown
selections retain their uncertainty. This paired calculation is implemented and
tested but has not yet produced a real auxiliary-model result.

## 5. Experiments and Results

### 5.1 SDD: a useful comparator is not a new neural contribution

| SDD comparison | ADE gain difference (pp) | Nominal 95% CI |
|---|---:|---:|
| Forest minus log neural (original primary) | +0.132728 | [-0.062063, +0.481336] |
| Forest minus log neural (same-count risk) | +0.723314 | [+0.610586, +0.827277] |
| Square minus log neural (strict primary) | +0.049515 | [-0.005267, +0.138640] |
| Square minus log neural (same-count risk) | -0.207277 | [-0.253252, -0.140711] |

**Table 1.** Earlier SDD development contrasts, four physical sites. Positive
values favour the first method. These are percentage-point differences in
site-relative ADE gain, not raw pixel errors. The original primary forest
comparison fails; the stronger same-count result is explicitly secondary.
Matching the empirical cost loss in a new neural head does not close the
same-count gap. All original eight policy rows, their easy degradation, missing
outcomes and tails remain in the
[SDD evidence table](../evidence_manuscript_v2/tables.md), rather than being selectively discarded.

A separate causal-only joint-support study found only six distinct recording
frames with changed decisions across the examined protected pools. It measured
opportunities, not predictive improvement. Consequently, this draft does not
claim a demonstrated joint-interaction contribution.

### 5.2 EuropeanSquares: fitting succeeds but selected protection does not

| ID | EuropeanSquares comparison | Change | Nominal 95% CI | Localities |
|---|---|---:|---:|---:|
| E1 | Cost minus original, TRAIN | -0.124240 | [-0.150777, -0.099481] | 12 |
| E2 | Cost minus original, validation | +0.104153 | [+0.018611, +0.220039] | 12 |
| E3 | Extension minus cost, validation | -0.126005 | [-0.246783, -0.030947] | 12 |
| E4 | Extension minus original, validation | -0.021852 | [-0.057900, -0.000175] | 12 |
| E5 | Extension minus additive, validation | +0.010294 | [-0.001532, +0.026280] | 12 |
| E6 | Temporal minus row-mean leaf, validation | -0.094342 | [-0.158824, -0.038614] | 12 |
| E7 | Temporal minus global temporal, validation | -0.137388 | [-0.232556, -0.047422] | 12 |
| E8 | Temporal minus row-mean leaf, complete labels | -0.098473 | [-0.159674, -0.047870] | 12 |
| E9 | Temporal minus row-mean leaf, original selected | +0.000043 | [-0.000140, +0.000256] | 11 |
| E10 | Extension minus cost, full utility | +0.000015 | [-0.000003, +0.000041] | 12 |
| E11 | Extension minus cost, matched utility | +0.000010 | [+0.000000, +0.000024] | 12 |

**Table 2.** Paired European development contrasts. E1-E5 use normalized
signed-score MSE; E6-E9 use normalized step-error MSE. Lower is better in both
families, but their magnitudes are not interchangeable. E10-E11 are utility
differences expressed as percent of full known reference cost; higher is better.
These utility contrasts are not ADE/FDE gains or a certificate for unknown
outcomes. The E9 legacy selected-cohort estimate has eleven supported localities;
the empty locality was not filled with zero. It is not a complete twelve-locality
selected-policy result. The newly registered readout instead fails a required
contrast with missing support.

All cost heads improve projected TRAIN fit, yet the validation contrast worsens
(E1 versus E2). The frozen diagnostic attributes about 82.15% of the error
increase to the easy-harm moment. This is an error decomposition, not evidence
that easy harm is the sole causal mechanism. Quality values outside TRAIN leaf
ranges are associated with most excess error; correlated diagnostic partitions
must not be added as independent causes.

The TRAIN-identity extension repairs global prediction error (E3-E4), but is not
clearly better than the additive control (E5). Its utility difference against
the cost parent lacks a strictly positive interval on either full or matched
readout (E10-E11). Most MSE improvement occurs outside the selected population.
This is why improved global fit does not justify advancing the policy.

| European policy | Selected occurrences | Unknown selected | Complete support / 72 | Defined easy risk / 72 | Known violations | Upper-bound violations | Worst easy upper % |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 95,455 | 918 | 33 | 43 | 4 | 7 | 5.4058 |
| additive | 112,456 | 1,143 | 19 | 61 | 20 | 42 | 1200.1684 |
| poisson | 111,031 | 1,050 | 37 | 51 | 2 | 11 | 18.0277 |
| cost | 96,720 | 926 | 36 | 48 | 4 | 7 | 5.7195 |
| extended | 96,718 | 926 | 36 | 47 | 4 | 7 | 5.7195 |

**Table 3.** All five retained European policies. Counts repeat across source
contexts and seeds; they are not independent people. Complete support and defined
easy-risk denominators are separate checks. The two-percent limit applies to
the risk ratio, not to classification accuracy. The extension retains four
known-label violations and seven upper-risk violations. Its worst defined easy
upper risk is 5.7195%, above the budget. Undefined heads do not count as passes.
No policy is promoted on the basis of this table.

### 5.3 Temporal information exists, selected-policy lift remains unproved

Temporal leaves outperform both controls on full validation signed step error
(E6-E7), and the complete-label sensitivity comparison also favours temporal
leaves (E8). This supports the preregistered information screen. It does not
establish better trajectory prediction: the forecasters are unchanged. It also
does not establish a better selector: the original-selected contrast (E9)
includes zero and has restricted support.

The original selected population contains 95,455 repeated row occurrences,
including 918 wholly unknown future outcomes. Within observed selected labels,
39.84% of pooled gross positive step harm cancels against negative step errors
on the same trajectory. This descriptive pooled calculation is neither a model
improvement percentage nor a cross-scene average. It explains why the registered
auxiliary target must remain distinct from the whole-trajectory primary target.

### 5.4 Registered neural experiment: not_run

The next experiment compares no auxiliary, repeated row-mean auxiliary and
stepwise auxiliary around the same width-32 moment encoder. It fixes primary
initialization, sampler draws, AdamW settings, a 2,000-update budget and final
checkpoint selection. Auxiliary weight is zero for the control and 0.1 for the
two supervised arms. The 72 source/head combinations yield 216 planned fits,
not 216 independent scenes. The readout retains all four strong controls.

Implementation and synthetic optimizer/resume tests are complete. A real TRAIN
batch passed forward/backward checks, but performed zero optimizer updates.
No real auxiliary checkpoint or new validation policy result exists at this
revision. Local storage admission failed and CREATE access was unavailable;
this is a resource blocker, not a negative scientific result. The registered
ten-GiB free-space reserve has not been weakened. Training will not be reported
as completed because a loss function or recovery path has passed a test.

## 6. Uncertainty and Failure Analysis

Intervals use 3,000 paired locality bootstrap draws in their respective studies.
Head seeds and dependent source contexts are averaged inside locality; overlapping
windows are not resampled as independent evidence. The four SDD sites and twelve
European localities are separate populations. Reported intervals are nominal
development uncertainty, not adjusted for the many preceding design choices.
Even an interval excluding zero cannot restore independent confirmation.

The evidence distinguishes four failure modes. First, optimization and
generalization diverge: successful TRAIN cost fitting does not preclude held
recording error growth. Second, prediction and selection diverge: large global
MSE repairs can leave selected-risk violations unchanged. Third, targets differ:
temporal positive error is not the original whole-trajectory harm. Fourth,
support is limited: some selected easy denominators are undefined and some
outcomes are unobserved. Increasing apparent sample size by reusing windows or
discarding unknown cases does not resolve these problems.

The studies do not identify a universally best estimator or prove that neural
cost learning cannot work. They rule out narrower claims about the examined
repairs. In particular, architecture names, successful gradient execution,
latent variance or an auxiliary prediction advantage are not substitutes for
positive controlled intervention utility with acceptable selected harm.

## 7. Limitations and Research Status

Independent calibration and confirmation remain absent from the results reported
here. Data exposure, partial visibility, detector tracking errors and historical
annotation interpolation restrict the inference contract. The present results
cannot be described as sensor-level causal perception or physical collision
avoidance. Scene-image, goal and interaction contributions are not established
by these cost-head studies, and useful joint decision support remains sparse.

The intervention is deterministic selection between frozen trajectories, not an
interactive environment model. M3W remains a 2.5D trajectory world-state research
scaffold, not true 3D or a foundation world model. No verified homography, metric
scale or effective-time conversion is used for these claims. Detector-derived
or inferred labels are not human gold. Stage5C has not been executed; SMC is off.

The most consequential missing evidence is a controlled positive policy result
that survives selected-risk and strong-comparator checks, followed by evaluation
under the reserved independent roles. An anonymous, venue-formatted submission,
full reproduction package and independent scientific review are also incomplete.
This manuscript is an evidence-bearing research draft, not a claim of CVPR or
CCF-A submission readiness.

## 8. Reproducibility

The [aggregate exporter](../../../scripts/build_m3w_evidence_manuscript_v3.py)
verifies eight SHA256-pinned inputs and regenerates all three tables, the European
contrast CSV and this manuscript. Table entries retain source paths and JSON
pointers. Rebuilding these artifacts is a fresh export of cached verified
results, not retraining, re-evaluation or a new bootstrap. Earlier immutable
reports preserve their original estimators, support rules and failed gates.

The [Chinese reproduction guide](reproduction_zh.md) distinguishes lightweight
document verification from private-data experimental reproduction. It links the
frozen training and readout entry points, storage guard and resume instructions.
Real training uses the native arm64 PyTorch environment, four compute threads
and zero DataLoader workers. Checkpoints preserve optimizer and random states;
they and third-party trajectories remain outside the public Git repository.
Public aggregate consistency does not verify private checkpoint contents.

## 9. Conclusion

Forecast quality, cost prediction and selected-policy reliability are different
empirical claims. Across the two development studies, conventional controls and
selected-cohort diagnostics expose failures hidden by average fitting scores.
Temporal labels contain useful predictive structure, but their value for safer
neural intervention is still an untested hypothesis. Establishing that link,
without replacing failed tests or weakening the risk definition, is the next
scientific requirement.

## References

1. Chenxin Xu, Robby T. Tan, Yuhong Tan, Siheng Chen, Yu Guang Wang, Xinchao Wang,
   and Yanfeng Wang. *EqMotion: Equivariant Multi-agent Motion Prediction with
   Invariant Interaction Reasoning.* CVPR, 2023.
   [Author manuscript](https://arxiv.org/abs/2303.10876v2).
2. Wenjie Luo, Cheolho Park, Andre Cornman, Benjamin Sapp, and Dragomir Anguelov.
   *JFP: Joint Future Prediction with Interactive Multi-Agent Modeling for
   Autonomous Driving.* arXiv:2212.08710, 2022.
   [Original manuscript](https://arxiv.org/abs/2212.08710).
3. Anqi Mao, Mehryar Mohri, and Yutao Zhong. *Regression with Multi-Expert Deferral.*
   ICML, PMLR 235:34738-34759, 2024.
   [Proceedings](https://proceedings.mlr.press/v235/mao24d.html).
4. Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candes, Michael I. Jordan,
   and Lihua Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve
   Risk Control.* arXiv:2110.01052, version 5, 2022.
   [Original manuscript](https://arxiv.org/abs/2110.01052v5).
5. Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and
   Tal Schuster. *Conformal Risk Control.* ICLR, 2024.
   [Proceedings paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).
6. Drew Prinster, Clara Fannjiang, Ji Won Park, Kyunghyun Cho, Anqi Liu,
   Suchi Saria, and Samuel Stanton. *Conformal Policy Control.*
   arXiv:2603.02196, version 1, 2026.
   [Version reviewed](https://arxiv.org/html/2603.02196v1).
7. Alexandre Robicquet, Amir Sadeghian, Alexandre Alahi, and Silvio Savarese.
   *Stanford Drone Dataset.*
   [Dataset release and associated ECCV 2016 work](https://cvgl.stanford.edu/projects/uav_data/).
8. Nils Wolff and Layne Perry. *Pedestrian Trajectory Dataset of Public European
   Squares.* Zenodo, release v1.1, 2026.
   [Dataset record](https://zenodo.org/records/18267205).

The locally audited release and use restrictions remain authoritative for the
cached assets; a link alone does not authorize redistribution.
