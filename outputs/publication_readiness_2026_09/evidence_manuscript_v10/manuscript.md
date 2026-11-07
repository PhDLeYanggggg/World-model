# When Should a Neural Forecast Replace a Motion Baseline?

**Working evidence draft. Previously exposed development data; not independent confirmation or submission-ready.**

## Abstract

Neural motion forecasts may improve average error while harming agents that a
causal baseline already predicts well. Choosing a forecast is therefore a
different problem from improving its mean accuracy. We investigate
baseline-relative intervention on moving-drone crowd trajectories with partial
future labels. The controller predicts benefit, positive harm, reference error
and label support from past-only inputs, then allocates substitutions jointly
within a query. We retain the original recording-level error and support
requirements, including undefined references. Matched experiments test source
support, training losses and past-image information without reusing new test
outcomes. Past-image cost heads improve fitting and label-availability
prediction, but their registered motion-versus-quality contrast remains
uncertain: a paired completion-lower difference of 0.462 percentage points has
an exploratory interval of [-1.095, 2.294]. The full-motion policy improves
available-label ADE by 5.867% over causal constant velocity yet passes only
12 of 72 required recording views. A fully labelled failure case exposes severe
harm underprediction. These results distinguish useful average routing from
reliable intervention; they do not establish a safe learned upgrade.
Conditioning the trajectory actor itself also fails its matched motion contrast (-1.804 percentage points on known labels). No new deployment or independent confirmation is established.

## 1. Introduction

A deployed forecasting system often already has a useful physical or analytic
motion predictor. The operational question is not simply whether a neural
network has lower average error. It is whether the network should replace
that predictor for this agent, in this query, without sacrificing the cases
that the baseline handles well. A favorable average can coexist with large
errors on simple trajectories or particular recordings.

Crowded moving-camera data make that distinction difficult. Apparent image
motion combines pedestrian movement and viewpoint changes. Tracks can lose
future labels through limited support or visibility. A future-informed oracle
can identify useful candidates without showing that past observations identify
them. A cost regressor can fit well while underestimating harm where its own
selection rule intervenes. An exact allocator cannot repair those wrong costs
merely by optimizing them more precisely.

We study these issues through a fixed, baseline-relative forecasting problem.
The action changes a forecast, not the pedestrian. Our empirical contract
separates positive harm from net improvement, requires support for all/easy
reference errors, and retains unknown outcomes rather than treating abstention
as a measured safe zero. We compare individual selection, a scene-query-wide
choice and joint choices with or without an intervention-count constraint.
The joint optimizer pools error budgets; it does not optimize pairwise
collisions or control a physical environment.

The contribution under investigation is reliable neural intervention under
partial labels, not a JEPA/Transformer combination or a new optical-flow
algorithm. The present native moving-drone evidence does not yet solve that
problem. We report matched positive component signals, failed whole-policy
checks, exact computational reproduction and the remaining model failure.
No independent confirmation or deployable world-model advance follows from
the experiments in this draft.

## 2. Related Work and Claim Scope

EqMotion separates geometric and pattern features and models agent
interactions [1]. We adapt a pinned author core with one head fixed before
training and evaluation. This is not a reproduction of its published training
recipe or best-of-20 scores. Our image-conditioned branch changes the input
to that core; its global geometric equivariance is not established merely by
reusing the original architecture.

Regression with multiple experts already includes joint and two-stage
predictor/deferral learning [2]. Continuous expert costs and choosing between
pretrained forecasts are therefore not new by themselves. The question here
is the empirical reliability of selected positive-harm/reference costs across
source-separated crowd recordings and incomplete future support.

Conformal Risk Control gives expected-risk guarantees for its specified
exchangeable, bounded-above, monotone-loss setting [3]. Our selected-reference
ratios are not automatically monotone or bounded: removing a low-harm
substitution can reduce a denominator and increase the ratio. No conformal
or population guarantee is claimed for the learned controller. Conservative
completion accounting is an error bound, not independent calibration.

MovingDroneCrowd++ provides moving-drone crowd images and head boxes with
tracking identities [4]. Its counting/tracking release is the data source;
the source-grouped 8/12 forecasting protocol below is a registered research
adaptation, not the publisher's official forecasting leaderboard. Background
motion measurement and simple feature conditioning are conventional tools.
Their inclusion would require a controlled downstream contribution, not just
a lower training loss.

Explicit separation of pedestrian and camera motion is also established
prior work. Neumann and Vedaldi normalize the viewpoint and predict future
ego motion for monocular pedestrian forecasting [5]. This motivates testing
geometric conditioning, not claiming camera compensation itself as new.
Their learned future-pose estimator is not implemented by our fixed affine
proxy; neither their reported benchmarks nor physical geometry transfer here.

Selective regression already shows that reducing coverage can worsen
subgroup error [6]. Multi-expert classification surrogates can also underfit
despite favorable consistency properties [7]. These observations are not new
claims of this project; their conditions do not establish our relative-ADE
recording contract. Weighted non-exchangeable risk control still requires
its specified monotone, bounded losses [8], not arbitrary selected ratios.

Trajectory certification under noisy observations [9] and joint conformal
regions under agents' policy changes [10] address different problems. Our
decision changes a forecast, not the pedestrians' behavior. Neither input-
robustness certification nor off-policy coverage is reproduced or inherited.

## 3. Forecast Substitution with Partial Labels

At a query, target-eligible agents have eight observed positions and a fixed
twelve-step request. Eligibility and neighbor support depend only on the past.
The current model uses at most eight complete, time-aligned nearest-neighbor
histories per target; it does not claim to represent every visible agent.
Positions remain in native image coordinates. The numerical time embedding
is relative to the request horizon, not a verified number of seconds.

Let b be the causal constant-velocity floor and p_k a candidate trajectory.
For a nonempty future observation mask M, define available-label error as

$$
c_M(p,y)=\frac{1}{|M|}\sum_{t\in M}\|p_t-y_t\|_2.
$$

The controller learns five nonnegative supervised moments on observed rows:

$$
\begin{aligned}
B_k&=[c_M(b,y)-c_M(p_k,y)]_+,\\
H_k&=[c_M(p_k,y)-c_M(b,y)]_+.
\end{aligned}
$$

$$
R=c_M(b,y),\qquad ER=E R,\qquad EH_k=E H_k.
$$

E denotes a positive baseline error below a cutoff fixed on producer-fitting
data. It is a future-defined evaluation/training label, never an inference
input. Positive harm does not cancel benefit from another agent. For the
selected set in each recording, the original empirical requirements are

$$
\frac{\sum H_k}{\sum R}\leq0.02,\qquad
\frac{\sum EH_k}{\sum ER}\leq0.02.
$$

The signed net easy-error increase must also be at most 2% over its original
full easy reference. All required references must be finite and supported.
An empty selected reference remains not estimable. The unchanged contract
contains no analytic identity-fallback exception or epsilon denominator.

For any two forecasts, the reverse triangle inequality gives the elementary
bound

$$
|c_M(p,y)-c_M(q,y)|\leq\max_t\|p_t-q_t\|_2=D(p,q).
$$

Thus a wholly unknown selected row contributes no asserted benefit and up to
its causal path disagreement to positive harm. A paired policy contrast uses
disagreement between the two policies on unknown rows, not subtraction of
separate lower bounds. Its denominator remains known floor-error mass. This
does not identify the actual unknown error, fill missing steps of a partial
track or turn available-label ADE into complete twelve-step ADE. The bound
is simple accounting, not a new theoretical contribution.

## 4. Models and Decisions

### 4.1 Frozen Forecasts and Joint Decisions

The existing causal bank includes constant position, finite-difference
constant velocity, two velocity decays and two past-only OLS velocity fits.
The learned forecast is a deterministic fixed-head EqMotion adaptation.
Current policy comparisons keep the same forecast bank, floor and six
nonzero neural interpolation amplitudes. Candidate-specific rollout diagnostics
are causal predictions, not observed future endpoints.

A cost head predicts conditional benefit/harm/reference moments and future
label-observability probability. Unknown-label contributions retain the
disagreement envelope. Reference and support branches see only candidate-
invariant inputs; changing a candidate cannot change their input reference.
The same architecture and original role-separated objective are used across
the image-cost ablation arms.

Independent admission requires positive predicted net benefit and both
predicted 2% moment budgets. Joint admission maximizes predicted total net
benefit with at most one candidate per agent, under the same two budgets.
The matched-count version retains the independent policy's exact count in
every query. The unrestricted version may change that count. A whole-query
uniform-candidate control and the frozen proposal-only control are reported
as well. Matched counts compare utility only; an empty control is not a
safety pass.

The mixed-integer solver has the original one-second per-query boundary.
Nonoptimal or numerically infeasible results retain a feasible parent. Exact
predicted feasibility and count checks are repeated on frozen choices.
Those checks are computational properties, not guarantees about actual harm.

![Training/evaluation labels are separate from causal model and allocator inputs. Joint allocation pools predicted error budgets, not physical collision constraints.](decision_pipeline.pdf)

### 4.2 Past-Image Cost Ablation

Sparse past-image tracking and robust partial-affine fitting measure a
background-motion proxy. Seven observed transitions ending at the query are
used. Head masks come from observed frames only. Unsupported measurements
remain explicit, not certified zero camera motion. Native image dimensions
provide boundary features, not walkability, homography or physical scale.

Four cost arms share 19,814 nominal parameters, initialization, draws,
source-separated training and 4,096 updates: no image; image bounds;
quality/geometry with transforms suppressed to identity; and full measured
motion/geometry. The registered primary is motion minus quality under the
joint-free decision. Bounds versus no-image is secondary. Full motion versus
no-image is a bundle contrast, not pure motion attribution.

All heads train on producer-A OOF donors only. They never fit own-B targets.
Each producer support excludes its corresponding own-B prediction sources.
Input normalizers use fitting rows only. We freeze all 36 choice contexts
before the complete own-B outcome comparison; no threshold is chosen from
that comparison. Three risk-head seeds are not three forecaster seeds.

### 4.3 Upstream Actor Repair

The first repair changes the actual forecaster's loss, keeping inputs,
architecture, draws and budget fixed. A prefix-balanced objective gives half
weight to available twelve-step ADE and half to supported 1/3/6-step errors.
Six source-held actors and a full old-control refit complete 4,000 updates
each. Rows with only late labels keep full loss; unknown rows remain in the
sampler without being assigned zero-error targets.

The next repair retains the original whole-twelve objective and conditions
the full trajectory network on past-image information. A bias-free 46-to-32-
to-8 branch adds a zero-initialized bounded offset to valid nodes' history
feature inputs. It does not add a trajectory residual decoder. It uses the
40 action-invariant context slots and six invariant-to-candidate image-boundary
slots; candidate-specific rollout fields are excluded. The offset is bounded
in latent space, not in trajectory error. Fitting-only normalizers balance
sources and never use donor or own-B statistics.

Quality and full-motion actors share 3,028,996 nominal and 582,404 trainable
parameters, the six existing source exclusions, the original sampler and
4,000 updates. Unused author heads remain frozen. One complete masked-context
control reproduces original shared-core training; two full treatment refits
check the first fixed design. These are not twelve complete exact refits.

### 4.4 Explicit Viewpoint Conditioning

A separate conventional repair aligns eight observed positions and their
time-aligned neighbors to the query image using seven past affine proxies.
All seven pairs must satisfy the pre-existing measurement support rule;
otherwise the full window retains identity imputation without deleting rows.
Of 125,540 windows, 94,242 have complete support and 31,298 retain identity
imputation. These populations are not interchangeable experimental cohorts.

The unchanged neural core predicts in that query frame. Fixed powers of the
last observed affine restore predicted native image coordinates before the
same whole-twelve loss. No actual future image or camera pose is used. This
extrapolation can fail under changing viewpoint and parallax. There are no new
learned parameters: 3,027,268 nominal and 580,676 trainable parameters remain.
A matched camera-compensated CV forecast tests geometry without neural learning.
An unsupported window is not frozen raw-model fallback: newly learned weights
can still change its prediction. All six actors and six camera-CV banks must
freeze before diagnostic scoring. The primary remains camera actor minus the
cached identity actor; secondary controls cannot replace it after readout.

## 5. Data Roles and Experimental Protocol

The current task uses the pinned MovingDroneCrowd++ archive at native
annotation stride one. The producer-A donor bank contains 125,540 past-
eligible agent windows, 14 source components and 34 recordings. The existing
own-B development cohort contains 62,466 windows, eight components and
24 recordings. Roles rotate between contexts and globally overlap; adding
their counts does not produce independent sources or independent test data.
Each fitted pipeline enforces its own source exclusions and teacher lineage.

Producer-A outcomes supervise source-excluded donor forecasting and cost
training. Own-B is already-exposed model-development evidence. No new nested-C,
outer-H, reserved moving-scene, DUT or DroneCrowd outcome is opened for the
experiments in this draft. Prior exposure cannot be undone by renaming a
cohort. Independent confirmation and the original broader main evaluation
remain separate unmet requirements.

All windows with complete past support remain, including partial and wholly
unknown futures. Available-label ADE and known endpoint diagnostics use their
own support; neither is relabelled complete ADE/FDE. Query-balanced source
statistics prevent large crowds from defining the independent sample size.
The image-cost primary uses 3,000 exploratory bootstrap draws over eight
source components after risk-seed averaging. The actor comparison uses the
registered source-level bootstrap over 14 donors. Overlapping windows and
repeated risk seeds are not independent replicates. These intervals do not
adjust for project-wide adaptive development.

Earlier European-square experiments use a different source protocol with
an observation sampling stride of twelve raw frames. Historical SDD studies
use pixel raw-frame horizons. Neither their absolute errors nor their gains
are pooled into the native-stride-one tables here. All datasets remain without
the verified time/geometry evidence needed for a new metric or seconds claim.

## 6. Results

### Table 1. Past-image cost heads on the full exposed own-B cohort

| Inputs | ADE gain vs CV (%) | Hard gain (%) | Easy gain (%) | Original supported views | Worst signed easy loss (%) |
|---|---:|---:|---:|---:|---:|
| No image | 0.700 | 0.638 | -0.308 | 18/72 | 12.41 |
| Bounds | 1.261 | 2.228 | -0.612 | 26/72 | 9.55 |
| Quality / geometry | 5.290 | 8.274 | -1.928 | 12/72 | 47.35 |
| Motion / geometry | 5.867 | 9.288 | -6.520 | 12/72 | 118.26 |

The 72 views repeat 24 recordings over three risk seeds. They are not
72 independent scenes. Mean easy gain is not the selected positive-harm
ratio or the worst recording's signed easy-net loss. Error/reference ratios
can exceed 100%; they are not probabilities or collision risks.

The registered motion-minus-quality paired completion-lower effect is
0.461541 percentage points with interval [-1.094911, 2.293574]. All three
risk-seed means are positive, but only 5 of eight source means are
positive and the interval crosses zero. The component screen fails. Bounds-
only and full-bundle intervals also cross zero. The motion policy has 43
known-easy positive-harm violations, 26 signed-easy violations and 9
undefined easy-reference views. Its full original recording contract fails.

All six motion/quality pairs improve final fitting-minibatch objective.
However, held-source conditional harm MSE rises from 0.001455 to 0.003445
and easy-harm MSE from 0.000328 to 0.000608 in common past-history-scale
units. Observability Brier improves from 0.049945 to 0.038765. Thus better
label-availability prediction is not accurate forecast-harm prediction.
These are empirical diagnostics, not conditional calibration guarantees.

![Known-label utility and unchanged original recording support for all four image-cost arms. Intervals are exploratory source-cluster intervals, not independent confirmation.](visual_cost_evidence.pdf)

### 6.1 Earlier Matched Repairs

The source-support/global-past-motion study completes 96 cost heads and
96 full exact refits. Its registered augmented global-minus-local paired
lower effect is 0.199 points with interval [-0.820, 1.254]. The interval
crosses zero, and the full original contract fails. Its old 50% A/B support
regime differs from the image study's A-only training. An unmatched score
difference between those studies cannot isolate an image effect.

The prefix-balanced actor repair improves paired known 1/3/6-step differences
by 12.439/2.915/0.677 points, but whole-twelve gain changes by only 0.098
points, with six of fourteen positive sources and a negative second-fold
mean. The registered paired lower is -9.474 points, interval
[-17.809, -3.083]. All 34 new raw-actor recording diagnostics still exceed
2% known-easy positive harm. This is a failed whole-forecast repair,
not success inferred from a favorable early prefix.

### 6.2 Image-Conditioned Actor Comparison

All twelve actors complete 4,000 updates, with both fixed-design full refits exact. The primary whole-twelve known difference is -1.804 percentage points; paired completion lower is -10.931, with exploratory source interval [-17.826, -5.288]. 7 of 14 source known contrasts are positive. The registered component progression screen is not supported. This remains a one-seed, previously exposed A-side component experiment, not a complete policy risk result or independent confirmation. Deployment is unchanged.

### Table 2. Source-balanced actor differences on exposed producer-A donors

| Contrast | Prefix | Known difference (pp) | Completion lower (pp) |
|---|---:|---:|---:|
| Motion - quality | 1 | -2.536 | -7.165 |
| Motion - quality | 3 | -1.417 | -5.970 |
| Motion - quality | 6 | -1.760 | -8.421 |
| Motion - quality | 12 | -1.804 | -10.931 |
| Quality - raw | 1 | +4.878 | -0.496 |
| Quality - raw | 3 | +1.034 | -4.426 |
| Quality - raw | 6 | +0.281 | -7.151 |
| Quality - raw | 12 | +0.737 | -12.363 |
| Motion - raw | 1 | +2.343 | -2.812 |
| Motion - raw | 3 | -0.382 | -6.337 |
| Motion - raw | 6 | -1.478 | -9.532 |
| Motion - raw | 12 | -1.066 | -15.021 |

The registered primary is motion minus quality over twelve steps.
Raw comparisons are secondary bundle contrasts. The quality bundle has
a small positive known difference, not a positive completion bound.
Neither an early-prefix signal nor a secondary bundle replaces the
failed primary. These are fourteen exposed development sources, not
independent confirmation or three forecaster seeds.

### 6.3 Explicit Query-View Geometry

Six new actors complete the same 4,000 updates with no additional learned
parameters. The full identity control and one fixed-design treatment
refit match exactly; all twelve actor/camera-CV banks and reported
scalars replay exactly. This is not six complete treatment refits.

The primary camera-actor minus identity known effect is -5.133
percentage points. Its completion lower is -21.447
with exploratory source interval [-33.682, -9.911].
3 of fourteen known source contrasts are positive;
the registered component screen is not supported.

### Table 3. Query-view controls on exposed producer-A donors

| Contrast | Prefix | Known difference (pp) | Completion lower (pp) |
|---|---:|---:|---:|
| Camera actor - identity | 1 | -1.125 | -8.444 |
| Camera actor - identity | 3 | -5.263 | -13.355 |
| Camera actor - identity | 6 | -6.351 | -17.333 |
| Camera actor - identity | 12 | -5.133 | -21.447 |
| Camera CV - raw CV | 1 | +0.544 | +0.057 |
| Camera CV - raw CV | 3 | +1.696 | +0.251 |
| Camera CV - raw CV | 6 | +2.584 | -0.175 |
| Camera CV - raw CV | 12 | +2.828 | -2.611 |
| Actor - camera CV | 1 | -15.352 | -21.683 |
| Actor - camera CV | 3 | -0.276 | -9.985 |
| Actor - camera CV | 6 | +4.674 | -9.191 |
| Actor - camera CV | 12 | +7.417 | -12.050 |

The simple camera-CV control isolates whether geometric extrapolation
alone is useful; its gain is not neural added value. All seven fixed
causal baselines and all fourteen source components remain in the
experiment report. No outcome-selected baseline replaces the CV floor.
This remains a one-seed exposed-development component comparison, not
an independent test or complete selected-policy risk certificate.
No new controller or deployment follows from this table.

The raw camera actor exceeds 2% known-easy positive harm in 33
of 34 recordings; this is a raw-actor diagnostic, not a full policy
certificate. In 6 of six matched designs its final fitting-
minibatch loss is also higher. That single-minibatch comparison is not
a training-population average or proof of underfitting. Past alignment
and future-affine restoration remain a bundled intervention; their
separate effects are not identified by this experiment.

### 6.4 Additional Observed-View Node Features

The replacement retains native geometry and adds aligned motion as
per-node features. A capacity-matched control suppresses the camera
transform. Both add 1,792 parameters, not a trajectory residual or
future camera-restoration operator. Twelve full 4,000-update fits,
one full shared-core control and two fixed-design full refits complete.
All twelve new predictions and reported statistics replay exactly.

### Table 4. Whole-12 dual-view contrasts on exposed donors

| Comparison | Known (pp) | Lower (pp) | Exploratory interval for lower |
|---|---:|---:|---|
| Camera - native node | +0.828 | -4.979 | [-7.692, -2.333] |
| Camera - original | -0.321 | -5.995 | [-8.461, -3.530] |

The matched comparison has 10/14 positive known sources;
the original-actor comparison has 6/14.
Both comparisons must pass. The registered component screen is not supported.
Raw known-easy positive harm exceeds 2% in 33 of 34 recordings;
this is not a complete selected-policy certificate. The reported lower
contrast bounds wholly unknown row contributions with the observed
reference fixed, not the full-grid error of partially labelled rows.
This is one forecaster seed and no new independent test or deployment.

### 6.5 Matched-Count Pair Decisions

On the same exposed forecast bank, we hold the switch count fixed per query and add an analytic pair term to source-excluded marginal costs. Across 125,540 rows, the reduction in squared normalized pair error is -0.000172 [-0.001445, +0.000907]. The component fails. A label-aware oracle adds only 0.000634 over the additive oracle; this is not learnable gain. Matching count does not match the original 2% risk budget.


In a separate three-seed predictor control, EqMotion minus PCT yields +55.02 pp on known labels but a completion lower bound of +14.46 pp [-8.19, +38.93] across 23 supported development groups. The 24-group primary remains not estimable. This is neither a three-seed intervention policy nor independent confirmation.

With three actual EqMotion seeds on exposed development sources, we also hold candidate paths and switch counts fixed and add between-seed dispersion to source-excluded boosted gain/harm models. Average cost error falls, but known utility changes only +0.049 pp and the completion-lower contrast is -3.535 pp [-8.981, -0.356]. The matched-count component fails; the full risk policy is not evaluated.

Against the fitting-selected OLS4 baseline, EqMotion gains +20.47 pp on known labels; the completion lower is -1.50 pp [-21.07, +14.22]. Reusing the original CV risk contract on all 34 recordings and three seeds, the learned missingness anchor passes 0/102 views. Easy completion risk exceeds 2% in 85 views and is undefined in 17. These repeated views are not independent evidence.

An offline label-aware ceiling finds 91/102 feasible binary views at free count and 90/102 at the old fixed count. Adding PCT and the other causal baselines supplies 101/102 feasible views, still not the whole roster. This isolates missing candidate support on these banks without demonstrating a learnable selector.

PCT easy-harm training reduces known harm by 6.98 pp but yields a paired lower of -7.68 pp [-16.96, -2.09]. Neither new trajectory bank recovers the unsupported view. With CV agreement training, completion lower versus CV is -40.03 pp and raw risk passes 0/102 views. Direct velocity integration raises oracle feasibility to 102/102, but neither raw variant passes any view. Peer context changes the paired lower by -7.71 pp [-12.51, -3.12]. At matched counts, the mixed term versus its marginal control changes the lower by -0.061 pp [-0.157, -0.006]. Filtering all actions before ranking gives +0.077 pp versus CV but passes only 2/102 original risk views. Nested source calibration passes 19/102 by retaining 474 of 6,184 switches, with no clear CV gain. Matched harm heads improve prediction error but keep 2/102 risk passes. With true outcomes and the same counts, only 30/102 screened views are feasible, versus at least 72/102 in the full bank. This is an action-space ceiling. Carrying predicted slack forward causally improves the matched lower by +0.017 pp [+0.002, +0.033] but passes 3/102. The full bank gains +0.014 pp over a reoptimized physical bank at common counts, reducing passes from 12 to 6. Predicted earning queries show realized net deficits for both budgets; all 42 donor fits retain zero credit. Adding half and quarter alternatives improves matched utility by +0.016 pp [+0.004, +0.028], but passes 0/102. Uniform quarter displacement of the original choices passes 29/102 at lower utility. Neither policy satisfies the original risk rule.

## 7. Failure Analysis and Limitations

Posthoc frozen-checkpoint diagnostics find all twelve final image projections nonzero (True). The largest known loss is source 067, where both arms have 0.000% and 0.000% clipped input elements, respectively. Thus neither an entirely inactive branch nor input clipping alone explains that case. These diagnostics are not retrained ablations or proof of a single causal failure mechanism.

The source-held error and selected policy answer different questions.
The past-image branch improves its fitted objective, but the controller
intervenes more often and continues to underestimate harm on selected
easy cases. In the posthoc worst signed-easy recording, frame 11 of
moving_scene_34__2, the quality control retains CV with ADE 5.2922 pixels.
Motion chooses the full neural forecast with ADE 96.4910 while predicting
harm 0.3323. All twelve future labels exist. This failure is not solely an
unknown-label completion penalty. It is a selected tail example, not
representative sampling or a stand-alone pointwise 2% budget test.

The image measurement can fail under weak texture, parallax or independently
moving background content. Native bounds do not reveal exits or future
visibility. The affine proxy is not calibrated camera pose, and the small
context head does not learn semantic pixels. The current experiments cannot
identify whether image semantics, better candidate dynamics, additional
independent training sources or better conditional uncertainty is the most
effective remaining repair.

A synthetic check also exposes a specific defect of removing only the future
view restoration. For observed native positions $h_t=p_t+c_t$, exact alignment
and current-origin removal can leave $p_t-p_0$ at the core, while native future
labels still depend on $c_{\mathrm{future}}-c_0$. Different observed camera motions can
therefore collapse to the same core representation with different labels.
This conventional information-loss example is not a new impossibility theorem
or proof of the combined actor's real-data failure. The incomplete one-sided
follow-up was stopped before outcome scoring. The replacement retains both
native geometry and aligned node features, which repairs this representation
defect without itself establishing a predictive or policy gain.

These source-excluded results use previously exposed data. Replay proves
repeatability, not independent confirmation; the single-seed actor cannot
replace the intended three-seed comparison. We retain all views, including
empty ones. Mean gains do not resolve the failed risk gate.

This is a 2D image-coordinate study, not a 3D, foundation, calibrated
physical or cross-dataset result. Stage5C and SMC remain disabled. A method
contribution still requires useful gains under the original contract and
independent confirmation.

## 8. Reproducibility and Responsible Use

Definitions, input schemas, source exclusions and comparison endpoints are
committed before fitting. Registrations bind source, feature and split hashes.
Model checkpoints retain optimizer, sampler/RNG, draws and logs. Native arm64
CPU execution uses four compute threads, one inter-op thread and zero loader
workers. Genuine interrupted-versus-continuous training is checked; NumPy
fallback and import-only tests are not counted as neural training.

For the actor study, preparation opens past-only arrays. Later generic loading
opens and hashes the complete packed training bank, including labels, for
integrity checks; only causal geometry and image context reach inference.
Forecast freeze precedes diagnostic scoring, not all target-file byte access.

The image-cost study includes 24 fits, 24 exact refits and prediction/scalar
replay. Actor refits are counted separately. Receipts and reconstruction
commands distinguish fresh runs, verified reuse and unrun work. Positive
and negative source slices are retained. Data, image/feature caches,
third-party code and weights stay outside Git. An anonymous end-to-end
training and prediction package remains outstanding.

A separate 6-file, 253,746-byte aggregate artifact replays the three completed actor primaries in an isolated temporary directory, without site packages, network access or the project workspace. It reconstructs fourteen-source means and the original fixed bootstrap draws, retaining their distinct seeds and negative results. This is standard-library reporting reproducibility, not new independent evidence, guaranteed anonymity or model-training/prediction reproduction.

An isolated ledger bundle also reconstructs 21 contrasts, 13 policy risk
summaries and the credit accounting from identity-stripped aggregates.

Crowd imagery and trajectory identities should be handled under the dataset
license and without attempting personal identification. We release no new
human identity labels or human-gold scene annotations. A future inference
system should not interpret error-relative checks as physical control safety.
The author is responsible for the claims and decision to submit.

## LLM Usage Statement

OpenAI Codex assisted implementation, experiment orchestration, debugging,
analysis and figure production in Methods and Experiments, and drafting
and revision throughout the manuscript. It did not supply trajectory ground
truth or independent validation. Reported results are tied to saved artifacts
and documented checks. The author is responsible for validating experiments,
citations, data-role boundaries and the final manuscript. Automated checks
do not replace scientific verification.

## References

1. Chenxin Xu, Robby T. Tan, Yuhong Tan, Siheng Chen, Yu Guang Wang,
   Xinchao Wang, and Yanfeng Wang. EqMotion: Equivariant Multi-Agent Motion
   Prediction with Invariant Interaction Reasoning. CVPR, 1410-1420, 2023.
   https://arxiv.org/abs/2303.10876

2. Anqi Mao, Mehryar Mohri, and Yutao Zhong. Regression with Multi-Expert
   Deferral. ICML, PMLR 235:34738-34759, 2024.
   https://proceedings.mlr.press/v235/mao24d.html

3. Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and
   Tal Schuster. Conformal Risk Control. ICLR, 2024.
   https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html

4. Yaowu Fan, Jia Wan, Tao Han, Andy J. Ma, Wanli Ouyang, and Antoni B.
   Chan. Video Individual Counting and Tracking from Moving Drones:
   A Benchmark and Methods. arXiv:2601.12500, 2026.
   Dataset release: https://huggingface.co/datasets/fyw1999/MovingDroneCrowd

5. Lukas Neumann and Andrea Vedaldi. Pedestrian and Ego-Vehicle Trajectory
   Prediction From Monocular Camera. CVPR, 10204-10212, 2021.
   https://openaccess.thecvf.com/content/CVPR2021/html/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.html

6. Abhin Shah, Yuheng Bu, Joshua K. Lee, Subhro Das, Rameswar Panda,
   Prasanna Sattigeri, and Gregory W. Wornell. Selective Regression under
   Fairness Criteria. ICML, PMLR 162:19598-19615, 2022.
   https://proceedings.mlr.press/v162/shah22a.html

7. Shuqi Liu, Yuzhou Cao, Lei Feng, Bo An, and Luke Ong. When More Experts
   Hurt: Underfitting in Multi-Expert Learning to Defer. ICML,
   PMLR 306:75507-75531, 2026.
   https://proceedings.mlr.press/v306/liu26g.html

8. Antonio Farinhas, Chrysoula Zerva, Dennis Ulmer, and Andre F. T. Martins.
   Non-Exchangeable Conformal Risk Control. ICLR, 2024.
   https://proceedings.iclr.cc/paper_files/paper/2024/hash/de04896f011beff76c91e094f72727f4-Abstract-Conference.html

9. Mohammadhossein Bahari, Saeed Saadatnejad, Amirhossein Askari Farsangi,
   Seyed-Mohsen Moosavi-Dezfooli, and Alexandre Alahi. Certified Human
   Trajectory Prediction. CVPR, 12301-12311, 2025.
   https://openaccess.thecvf.com/content/CVPR2025/html/Bahari_Certified_Human_Trajectory_Prediction_CVPR_2025_paper.html

10. Tom Kuipers, Renukanandan Tumu, Shuo Yang, Milad Kazemi, Rahul Mangharam,
    and Nicola Paoletti. Conformal Off-Policy Prediction for Multi-Agent
    Systems. IEEE CDC, 2024.
    https://arxiv.org/abs/2403.16871
