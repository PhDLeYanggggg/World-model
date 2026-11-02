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
rankings. At the time this section is added,training is in progress;no policy
improvement is asserted. This is a controlled learning change,not a new theorem.

An event probability does not identify expected harm. For fixed R=1,both a
1%chance of H=1 and a1%chance of H=100 give the same probability of H>0.02R,
but expected harms0.01and1differ by100x. A cost-tied BCE logit therefore need not
remain a conditional mean cost estimate. The original magnitude losses and full
recording criteria must remain;auxiliary loss or sign accuracy cannot certify
their success. Future easy status is a loss label only,never an inference input.
