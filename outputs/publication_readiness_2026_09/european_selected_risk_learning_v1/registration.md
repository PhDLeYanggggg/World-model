# Selected-Risk Moment Learning and Scene-Query Allocation

## Material Passport
Registered before new training. Fresh risk-head fits and frozen source-C
readout, cached_verified fixed forecasts and prior cost heads. This is opened
source-development evidence, not independent calibration or confirmation.
The six opened selection localities are not evaluated this round. The separate
12 reserved calibration and six confirmation localities stay closed.

## Question
Does explicitly learning coherent all/easy cost moments and constraining their
means on causal proposal subsets reduce the selected-harm optimism observed
in the preceding calibration experiment? Does query-level allocation help
at the same nominal predicted-risk budget? Neither answer is assumed positive.

Earlier history/disagreement percentile rejection removed useful forecasts.
Do not repeat it as a deployment rule. Standardized-feature support bins at
B's 90th/99th percentiles are descriptive only; no support cutoff is tuned or
used for rejection here.

## Fixed Roles and Budget
Keep the three four-locality source rosters and all six ordered A/B pairs.
A trains the complete forecasting/floor chain. B trains the new cost models,
preprocessing, fixed component scales and proposal-subset supervision. C is
the remaining four-locality roster, excluded from both fitted chains, and is
used only for the new fixed readout. C was historically opened development;
this registration cannot make it independently unseen.

Three seeds 17/29/43; two full/motion-only forecast pairs; two learned arms.
All 72 small Torch heads receive 2,000 updates, batch256, width64, lr0.0003,
clip5, AdamW decay0.0001. A real 100-update pilot resumes within that budget.
No new forecaster, new test-driven hyperparameter choice or early stopping.
Keep input rows, known-label draws and preprocessing matched across the arms.

## Controlled Objectives
Predict D_all, H_all, D_easy, H_easy for the same delivered R/P pair. Easy
means positive CV error below A's fixed cut. Targets use future labels only
in the loss/readout. The four-output head enforces nonnegative moments,
H_all <= causal max-rollout envelope, D_easy <= D_all and H_easy <= H_all.
All inputs remain the fixed causal 383-feature bridge schema.

The mean arm uses squared residuals divided by each component's fixed B-only
root mean square, with a numerical 1e-4 floor in B CV-cost units. The selected
arm adds weight1 on squared group-mean standardized residuals. Groups are
each B locality crossed with population, frozen-utility-positive proposal and
frozen-neural-raw selected subsets. Membership is computed from past inputs
and fixed B-trained models, not from future easy labels. Teacher selection
on B is in-sample; C is excluded. This is not a claim of cross-fitted B labels.
Both arms otherwise share initialization, draws, scales and training budget.
Comparison against the earlier hurdle/ranking head is a package comparison;
only selected-vs-mean isolates the added group objective.

## Frozen Policies
For each pair keep reference, old neural and old ridge. Each new arm supplies:
all-risk individual gate; dual all/easy individual gate; scene-query uniform
proposal acceptance; query-greedy allocation. All retain positive old-neural
utility, current-motion and nonidentical-rollout prerequisites.

Individual gates use 0.02 predicted H/D. Query rules have separate all/easy
budgets of 0.02 times summed predicted reference mass over every indexed agent
at the same locality/recording/current frame, including unknown futures.
Uniform acceptance switches all eligible proposals only if both sums fit.
Greedy allocation orders positive predicted net utility, breaking ties by
stable row ID, and accepts a proposal only when both remaining budgets fit.
This is a fixed greedy allocation, not an optimal knapsack or collision solver.
An equal-per-query-count stable hash control tests selected-joint ordering;
its counts, not its risk, are matched. It is diagnostic, not a safe policy.
Zero/missing predicted event mass falls back. Actual risk is evaluated using
the unchanged delivered-R denominator and CV-defined easy event, never a
predicted surrogate substituted for the scientific constraint.

Freeze all 432 decision views and push before outcome readout. No threshold
refit. Publish all/easy/hard/complete ADE, FDE, tails, intervention, realized
positive harm, support-bin residuals, all negative arms and per-locality results.
Three-seed means within locality and3,000 locality resamples are exploratory:
each C readout has four localities and source-role views overlap. No window-level
independence, multiplicity-adjusted discovery or new risk guarantee is claimed.

## Decision and Resource Rules
If selected-vs-mean does not improve risk/accuracy, do not credit the new loss.
If joint only expands coverage without matched-count gain, do not claim better
ranking. If either violates easy or positive-harm constraints, do not deploy.
Positive source-only results still need a separately frozen development check,
independent calibration feasibility and confirmation. No best seed is selected.
Native arm64 CPU4, interop1, workers0; atomic checkpoints, resume, heartbeat,
PID and10GiB disk reserve. CREATE queue is checked read-only; no remote job
is needed unless the local measured cost justifies it. Do not touch other jobs.

## Literature and Claim Boundary
Subgroup calibration is established prior work, not a new M3W claim:
[Hebert-Johnson et al., ICML2018](https://proceedings.mlr.press/v80/hebert-johnson18a.html)
formalize calibration over computationally identified subsets;
[Globus-Harris et al., ICML2023](https://proceedings.mlr.press/v202/globus-harris23a.html)
connect multicalibration and squared-error regression. This finite-group neural
penalty does not implement or inherit their algorithms' guarantees.
Image-pixel obs8/pred12 annotation steps at raw stride12, detector-derived
labels. No metric/seconds, human-gold, physical-safety, true3D or foundation
claim. Stage5C and SMC stay off. Historical contaminated scores remain exploratory.
