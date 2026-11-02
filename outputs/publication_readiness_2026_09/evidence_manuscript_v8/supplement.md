# Technical Supplement: Partial Labels, Query Costs And Published-Loss Control

Working anonymous manuscript material. Elementary completion/support facts are
reused from the earlier technical supplement and multi-action methods appendix;
they are not new consistency or population-safety theorems. This revision updates
the algorithms,coverage and role descriptions to the current SOURCE studies.
It is not an independent experiment or a submission-readiness certificate.

## S1. Target And Causal Boundary

For history x,let b(x) be the frozen floor and p_k(x),k=1,...,7,the fixed candidate
forecasts. A decision replaces a forecast,not the physical agent's behavior.
On a nonempty observed future mask M,define

```text
c(p,y;M) = mean_{t in M} ||p_t-y_t||_2
R = c(b,y;M)
B_k = [R-c(p_k,y;M)]_+
H_k = [c(p_k,y;M)-R]_+
ER = E R; EH_k = E H_k.
```

E is the frozen producer-fitting definition of an easy event. Its realized value,
ground-truth future endpoints and future validity masks are loss/evaluation
labels,never inference inputs. Predicted baseline endpoints are causal and legal.
Unknown rows retain a missing label vector;they are not zero-error rows.
Known partial-mask costs stay fixed throughout completion accounting. This does
not silently change the estimand into fully observed twelve-step trajectory ADE.
An unavailable requested final endpoint leaves full-roster FDE not estimable.

## S2. Deterministic Completion Bound

Reverse triangle inequality gives,for every future point,

```text
abs(||p_t-y_t||-||b_t-y_t||) <= ||p_t-b_t||.
```

Hence any common nonempty-mask ADE difference is bounded by
D_k=max_t||p_kt-b_t||. The implementation can retain a larger pre-existing causal
CV disagreement envelope. Its conservatism must not be described as measured
hidden harm,or as an assertion that every bound extreme is physically attainable.

For fixed actions,unknown selected rows contribute at worst -D to utility and
at most D to positive harm. Retaining known positive reference mass in the
denominator yields a conservative ratio upper bound when that mass is positive.
A missing/zero selected denominator is not estimable. There is no denominator
epsilon that can turn it into evidence of safety without changing the quantity.

For two policies A and B,use their actual chosen trajectory disagreement:

```text
C = sum_known [c(p_A,y;M)-c(p_B,y;M)]
U = sum_unknown max_t ||p_A,t-p_B,t||_2
paired utility difference lies in [C-U,C+U].
```

This cancels identical unknown choices. Subtracting two individual utility lower
bounds is not the paired lower bound. There is no missing-at-random assumption,
outcome imputation or use of an evaluation mask in the controller.

## S3. Why Mean Utility Does Not Certify Selected Risk

Positive harm and benefit satisfy R-C=B-H. The net sum can improve while some
rows or recordings suffer large positive harm. For positive reference weights,
sum(H)/sum(R) is a reference-weighted average of row ratios;zero-reference
positive-harm rows can only worsen it. If every eligible positive-reference
action exceeds a tolerance,combining those actions cannot satisfy it. This is a
necessary-support observation,not a sufficient joint feasibility theorem.

Likewise,nested score thresholds need not yield monotone risk ratios. Removing a
low-harm row can shrink the reference denominator enough to increase the ratio.
The earlier synthetic example has successive risks10%,0.09901%,5%as a threshold
is relaxed. It is an algebraic check,not a real-data result or a proof that all
calibration is impossible. A standard monotone-loss guarantee cannot simply be
attached to this selected harm/reference ratio.

The current protocol requires the full policy on every original recording to
meet the all/easy positive-harm and easy-preservation requirements. Count-matched
controls measure utility,not safety. An empty required comparison stays not
estimable. No intervention cannot establish a positive intervention contribution.

## S4. Coherent Costs And Choice-Conditioned Training

The current critic's causal inputs include462frozen fields per action,predicted
cost anchors,disagreement envelopes,action codes and a fixed causal initial
selected-set context. Both training arms share31,119active weights and warm
states. A differentiable decoder shares R/ER over causally available candidates,
caps ER at R and B at R,and retains H/EH. This is algebraic coherence,not calibrated
conditional confidence or a claim that forecasts changed.

The fixed/own-choice comparison changes only the actions used in the aggregate
signed-risk residual. The own choices come from detached current predicted costs
and the unchanged native allocator,with the original two2%predicted constraints,
own independent count and feasible-parent fallback. The context input itself
stays fixed. This is stop-gradient choice-conditioned regression,not a derivative
through an integer optimizer or a published decision-consistency guarantee.

Only nonempty selected queries with every selected outcome known receive this
aggregate target. Other known legal actions retain pointwise cost supervision.
Unknown contextual rows remain in the causal computation. Empty/unknown query
targets are masked,not fabricated as zero. Complete cyclic sampling gives every
original query at least four visits. Monitor losses do not select checkpoints.

## S5. Current-Bank Published Deferral Adapter

The old-bank published deferral scores cannot be compared as if they used the
new neural forecast. The new adapter reuses the implemented two-stage logistic
loss from Mao et al.,ICML2024,Section4Eq3,on this exact updated bank and causal
schema. The floor plus seven forecasts form eight experts. With normalized
continuous costs d_j and logits f_j,the implemented loss is

```text
L_TDEF = -sum_j (sum_k d_k-d_j) log_softmax(f)_j / log(2).
```

There is no one-hot oracle class or inference-cost surcharge. The companion
control regresses those same d_j by softplus(f_j) squared error. The two score
networks share the same30,726active parameters,warm encoder,zero outputs and
sampling. This capacity is not identical to the31,119parameter critic. The floor
uses the same analytic hidden pool as the reference branch;unavailable actions
tie to its score/cost and cannot count as fake interventions.

Each query's loss averages known rows,then the16-query batch averages queries.
Wholly unknown queries have no supervised target. This is an application adapter
with query weighting,context and action aliases,not a reproduction of the original
experiments or verification of its bounded-loss/hypothesis assumptions.

Raw selection maximizes expert score;ADE selection equivalently minimizes the
expected-cost output. The guarded variant filters by the frozen own critic's
original pointwise predicted2%rules and positive gain. There is no threshold
refit,new calibration or conversion of arbitrary logits into native ADE utility.
New independent choices and the original frozen independent/joint controls are
evaluated under the same actual trajectory scores and recording requirements.

## S6. Scope,Statistics And Reproduction

All current comparisons retain149,510unique targets,74recordings,12exposed SOURCE
localities and72context/head/source views. Three cost seeds share forecaster43.
Earlier exposed inner-held results and updated-bank SOURCE fits are not a common
independent leaderboard. Bootstrap first averages repeated context/head views
within locality,then resamples localities3000times. Overlapping windows do not
create independent experimental units. These intervals are conditional source-
development evidence,not confirmation after research selection.

Both recent two-arm studies retain208,320primary updates plus14,336full-refit
updates. Their own controls require actual interruption/recovery and two exact
first-context full-budget refits,not just successful imports. Final verification
re-infers all new predictions and scores saved choices;it does not rerun a timed
integer solver. Public scalar report reconstruction is a separate operation from
training/inference and from independent scientific replication.

Use the per-study reproduction instructions and immutable registration hashes.
Checkpoints,raw trajectories,features and third-party files remain private.
The data are oblique detector-silver/local-image on obs8/pred12/rawstride12.
There is no metric,effective-second,human-gold,true-3D,foundation or physical-
safety claim. No Stage5C execution or SMC is performed. These materials remain
a working research package until useful independent evidence is established.

## S7. Post-Readout Fixed-Guard Diagnosis

A separately registered label-aware SOURCE diagnostic retains all72groups and
444recording/context/cost-seed views. It does not change inference,fit a threshold,
or redefine the primary after seeing its failure. Exact guard-bit probes reproduce
the seven-action eligibility mask before any outcome-based interpretation.

Among808,680repeated legal known easy action occurrences satisfying both true
signed-risk inequalities,664,755are rejected by the guard;620,297of these also
have strictly positive observed benefit. Conversely,75,663known signed-risk-unsafe
occurrences are admitted across all known rows. These counts repeat actions,
contexts and seeds;they are not independent samples or deployment confusion rates.
Observed signed-risk safety does not certify missing outcomes or physical safety.

Twelve views across four unique source/record pairs have positive full known easy
reference but no guarded known easy action. Their original legal banks contain
296true safe easy occurrences,208strictly beneficial. Thus a poor guard can remove
needed denominator anchors despite useful legal forecasts. For six repeated
views of one small source record,each of the three known easy rows belongs to a
singleton original target query. The unchanged native allocator cannot recover
known easy support by pooling a second target in those queries. The other blocked
views include multi-target queries;pointwise blockage does not prove their joint
allocation infeasible. Unknown outcomes cannot provide a known denominator for
this conservative certificate,but need not have zero realized easy reference.

These results motivate two-sided cost repair,not simply stricter filtering or
better ranking. They do not turn a label-aware witness into a causal model,
remove the failed primary,or show that missing-outcome bounds caused all errors.
The matched published-loss comparison remains5/9fitting gates with no deployment
promotion. No independent role is opened by this diagnosis.

## S8. Boundary-Event Supervision Is Not A Mean-Risk Guarantee

A current SOURCE control retains moment regression and adds two-sided BCE on
log harm versus detached predicted0.02-reference. Its two arms have matched
capacity,warm state,draws and complete budgets,and share frozen published-loss
rankings. All36fits,two full refits and72-group numerical replay now finish.
Known gain rises10.58%to21.43%,but supported recording-risk views fall20to0of444.
The primary lower effect is+4.83pp with CI[-3.07,9.34],so fitting advancement fails.
This is an observed routing gain,not a safe policy or a new theorem.

An event probability does not identify expected harm. For fixed R=1,both a
1%chance of H=1 and a1%chance of H=100 give the same probability of H>0.02R,
but expected harms0.01and1differ by100x. A cost-tied BCE logit therefore need not
remain a conditional mean cost estimate. The original magnitude losses and full
recording criteria must remain;auxiliary loss or sign accuracy cannot certify
their success. Future easy status is a loss label only,never an inference input.

For fixed budget c and q=P(H>c),the tied probability p(h)=h/(h+c) has expected
BCE derivative(p(h)-q)/h,ignoring the numerical epsilon. At h=E[H],the magnitude-
MSE derivative is zero but this auxiliary generally is not. Shared output tying
can therefore move a mean-cost estimator away from its proper target even with
unlimited capacity. This elementary observation is not a unique attribution of
the empirical error. It motivates separating event prediction from cost magnitude
semantics,not calling the tied sigmoid calibrated risk. The current boundary
arm improves its BCE while worsening the common anchor objective and known risk.

## S9. Conditional Easy Occurrence And Cost Magnitude

The next matched control reuses the earlier easy-hurdle mechanism on the current
full seven-action bank. Both arms have 31,381 active parameters, the same warm
causal encoder, equal fitting-prior output initialization, complete cyclic draws
and fixed-initial-choice query supervision. They are matched to each other, not
an exact prediction-preserving continuation of the preceding critic. The old
easy-hurdle negative result is retained rather than presented as a new method.

For causal X, predict q(X) and conditional costs B_e,H_e,R_e for e in {0,1}:

```text
B = (1-q) B_0 + q B_1
H = (1-q) H_0 + q H_1
R = (1-q) R_0 + q R_1
ER = q R_1; EH = q H_1.
```

The reference levels and q are shared across candidate actions. Conditional
benefit/harm fractions are bounded by causal path disagreement, and benefit is
also capped at its corresponding reference. The decoder therefore preserves
B<=R, ER<=R, EH<=H and B+H<=D. These are statistical cost factors, not future
trajectory branches, latent rollout or stochastic proposals.

The marginal arm regresses the five composed moments and fixed-query residual.
The identified arm adds unit-weight easy-occurrence BCE and unit-weight
class-balanced conditional B/H/R squared loss. Class weights and output priors
come only from fitting sources. Actual E is defined by the original future CV
cost and fixed easy cutoff, not ER>0: an easy row can have zero floor reference.
It is a loss/evaluation label only; strict causal inference rejects that field.
Wholly unknown outcome rows remain NaN and have no supervised contribution.

This separates occurrence semantics from a cost-tied boundary sigmoid. However,
for q>0, EH-0.02ER=q(H_1-0.02R_1), so a more accurate occurrence probability alone
does not change the pointwise easy-risk sign. Conditional magnitude estimates,
selected support and complete observed/unobserved risk still matter. The name
"identified" denotes the supervised arm, not identification of missing outcomes
or a causal-effect, calibration or population-safety guarantee.

The frozen TDEF ranking, forecast bank and all original recording requirements
are unchanged. Complete fit/readout/replay status and results are taken only
from the experiment's immutable scalar receipts; no partial loss is used to
promote a policy. This remains exposed SOURCE fitting, not independent evidence.

## S10. Frozen Forecast Readout On Model-Selection Localities

After all36 SOURCE-trained forecasts were frozen and separately replayed, the
first supporting outcome readout opened six model-selection localities. The
38,102 targets in28recordings are new to this forecast readout, not training.
These six localities are now exposed for model selection and must not later be
called untouched final confirmation. Risk calibration, confirmation and
DroneCrowd remain closed. No new fitting, normalization or threshold search was
performed. The main independent-versus-joint policy question is unchanged.

Three actual forecasting seeds17/29/43, three SOURCE folds and representations
disabled/current/noise/history produce54matched fold/seed/locality cells. Each
SOURCE producer's reference baseline and easy/hard thresholds remain fixed.
The input bridge uses the actual partial-neighbor observation mask, not generic
complete-neighbor context. Original SOURCE feature parity and a complete fresh
raw feature reparse pass; all36 forecasts and nine disabled-parent controls
replay exactly. A later separate reparse reproduces all28 label arrays and54
metric groups. Inference does not access target masks or future endpoints.

Known-mask ADE gain over the fixed producer floor is11.67% history,9.07%
disabled,8.52% current and8.42% noise. The paired history-minus-disabled known
effect is+2.60pp [95% locality bootstrap CI:1.73,3.81], positive in all six
localities. However, the registered paired completion-lower primary is-2.53pp
[-3.67,-1.49], negative in all six; the primary fails. Unknown-disagreement
completion is a bound under the registered partial-mask convention, not a
recovery of missing steps of partially labelled trajectories. Individual lower
bounds cannot be subtracted to replace this paired lower contrast.

There are37,254known targets,21,434complete12,15,820partial-but-known and848
wholly unknown targets. Available final endpoints number27,694; full-roster FDE
is not estimable. The roster is not selected by future completeness. Intervals
use3,000locality bootstrap resamples after averaging folds/seeds, not rows or
repeated views as independent samples. These are model-selection development
intervals, not independent final confirmation or multiplicity-adjusted evidence.

History full substitution supports0/252record-risk views, with249known-all
and247known-easy positive-harm violations. Worst known easy net degradation
is47.98%. Known harm independently fails safety; this is not only missing-label
uncertainty. Floor abstention has zero known harm but an undefined selected-easy
denominator, not proof of positive learned-intervention safety. The252views are
28recordings repeated for nine producers. Original2% requirements are unchanged.

Disabled has zero detector contribution within the same89,026total-parameter
architecture; its512detector parameters are unused, so active capacity is not
matched to history. Current/noise activate that same branch. Exploratory known
history-minus-current and history-minus-noise effects are+3.16pp [1.72,5.13]
and+3.25pp [2.33,4.46]; they do not replace the registered failed primary.

This supports known-label forecast-input utility, not a safe policy, scene-joint
decision method, JEPA contribution, independently confirmed dynamics advance
or submission readiness. The complete public scalar readout and failure report
are retained in european_selection_forecast_readout_v1. Local-image pixel,
obs8/pred12/rawstride12 and detector-silver limitations remain. No metric,
seconds-level, true3D, foundation, Stage5C or SMC claim follows.

## S11. SOURCE-To-Selection Cost Controller Comparison

Following S10,18matched real cost critics fit only the eight SOURCE localities
outside each frozen forecaster's four producer-fit sites. Three actual forecasting
seeds and three SOURCE folds remain; row/context critics share29,253parameters,
initialization, draws and complete four-visit query coverage. Inputs are380causal
geometry/rollout and64detector-history fields. Context adds only parameter-free
other-target pooling at the same frame. No selection labels train the critic,
normalizers or thresholds. This is a new cost fit, not new trajectory dynamics.

All18fits, two complete exact refits and18policy freezes finish before scoring.
Training uses141,312primary and14,336refit updates in541.08s locally,14.85GBpeak
RSS. The original tiny numerical recovery failure is retained; fixed-membership
reductions pass a new real recovery pilot and full exact refits without relaxed
tolerance. A missing private checkpoint directory is repaired operationally,
without a registered definition change. All18cost inferences and54frozen-action
metric groups later replay; solver optimization is not claimed reexecuted.

The primary is context matched-joint minus context independent paired completion
lower. It is+0.0016077pp [-0.0152463,+0.0168448]; the known secondary is+0.0137784pp
[-0.0080937,+0.0344373]. Both are inconclusive. Context-minus-row matched-joint
known effect is-0.0012360pp; its completion lower is-0.2111303pp with an entirely
negative interval. Context matched-joint known gain over its SOURCE floor is
0.1733%, but only21/252repeated record-risk views pass, with157known-all and156
known-easy violations and7.0157%worst known easy degradation. Original2%rules
remain. The36empty count-matched controls are not estimable,not safety passes.

The neural-action bank adds a small known+0.05748pp [0.03197,0.07749] over its
physical-action ablation, but completion lower includes zero and risk fails.
That ablation removes the action from the same critic; it is not retrained.
Raw neural known gain11.6729% does not establish that the critic can safely
identify its useful rows. A label-free causal-action audit shows19,496switches
in342,918repeated context row views; matched joint replaces801neural actions
with physical actions at the same total count. It is not a no-switch artifact.
These counts do not measure actual calibration or prove a unique failure cause.

This tests contemporaneous risk-budget pooling, not pairwise conflict/composition
optimization. All six selection localities remain model-selection exposed;
risk calibration, confirmation and DroneCrowd remain closed. Available-label
ADE, wholly unknown completion, detector-silver/pixel/raw-step limitations and
bootstrap unit are unchanged. This does not support deployment or submission
readiness. The complete readout, actual loss traces, negative results and causal
action diagnosis are retained in european_selection_controller_v1.
