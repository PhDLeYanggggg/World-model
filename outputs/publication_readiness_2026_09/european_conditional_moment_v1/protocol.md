# Coherent Conditional-Moment Identification On The Current Bank

## Question And Prior Reuse

The completed tied boundary-BCE control improves observed mean utility but
worsens recording risk. Its event logit generally distorts conditional-mean
cost semantics. The next comparison uses a separate event probability and
conditional cost levels whose mixture forms expected observed-cost moments.

This reuses the existing easy-hurdle occurrence/conditional supervision idea.
The earlier experiment already found better factor fitting without better
selection, so this is not presented as a new hurdle method. The new application
is the current full seven-action, five-moment coherent bank, shared reference
costs, causal selected-set context, frozen TDEF ranking and complete cyclic
coverage. Old negative results remain negative and are not comparable as if
they used this updated architecture/bank or identical training budget.

## Architecture And Causal Boundary

Both arms have31,381active weights: the same64-wide causal encoder, four
candidate-conditional logits and three shared outputs. The encoder is copied
from the same completed own-choice critic. New output weights are zero and
their biases use identical source-equal fitting-only priors in both arms.
This is not an exact prediction-preserving continuation of the previous31,119-
parameter critic; comparison is controlled within the new two-arm architecture.

E is the unchanged future CV-error easy event. It is a supervised label only.
From past-only context, the model predicts q=P(E|X) on the known-label fitting
cohort, conditional B/H/R levels for E=0 and E=1, and constructs:

```text
B = (1-q) B0 + q B1
H = (1-q) H0 + q H1
R = (1-q) R0 + q R1
ER = q R1
EH = q H1
```

The scalar event probability and both reference levels are shared across all
candidate actions for an agent. Conditional simplex costs obey B_e+H_e<=D;
benefit is capped at R_e. Hence B<=R,ER<=R,EH<=H and B+H<=D are coherent without
treating an event probability as a mean-cost ratio. The event factor does not
generate future paths, latent rollouts or stochastic proposals. No Stage5C/SMC.

Inference accepts only the existing causal schema, never realized E, a future
endpoint, future mask, central velocity, test goal or any future target. The
model predicts q; it does not know E. No q threshold is added to deployment.
Explicit E is derived from fitting CV labels and the already frozen easy cutoff,
not from ER>0, which would mislabel an easy row with zero floor-reference error.
Both ER/EH label identities are checked. Unknown rows retain NaN labels and
remain in causal context and sampling; none receives observed-target loss.

## Matched Objectives

Both arms use the same normalized five-moment squared loss and fixed-initial-
choice query signed-risk residual. Their shared anchor is half the sum of these
two terms, with all earlier RMS units frozen. The marginal arm optimizes only
this anchor, while measuring the auxiliaries as diagnostics.

The identified arm adds unit-weight BCE for E and conditional B/H/R squared
loss for the realized fitting stratum. This conditional loss has fixed class
weights0.5/P_fit(E=e), using the source-equal fitting prior only. It cannot
directly win by turning a mean-harm cost into an event probability. No new
severity threshold, readout-dependent weight, validation search or denominator
epsilon is introduced. Empty/unknown targets are masked, not supervised zero.

Under a correctly specified known-label conditional model, ordinary conditional
mean and occurrence targets are compatible by total expectation. This is an
elementary modeling fact, not a population-safety theorem, proof of empirical
calibration, identification of unobserved outcomes or missing-at-random claim.
Model misspecification, incomplete outcomes and induced-selection errors remain.

## Full Training And Readout

All36fits,208,320main updates,two exact first-context full-budget refits
(14,336additional updates),and an actual400-update interruption/recovery pilot
are required. Both arms keep the same initial state,cyclic draws,16-query
source-balanced batches,complete minimum-four-visit coverage,optimizer and
final fixed budgets. Checkpoints are not selected using any policy outcome.

Freeze all endpoints before evaluating all72original groups. Required controls
include marginal and identified guarded TDEF,the preceding moment and tied-BCE
guards,the original guarded TDEF,own-joint,raw TDEF,and floor. All forecasts,
TDEF scores,normalizers,149,510targets,74recordings and12SOURCE localities stay.
The all/easy2%pointwise prediction rules and every required recording's actual
risk/easy preservation requirements are unchanged. No future label selects an
inference row. Count matches compare utility only;empty comparisons are not
estimable,not safe zero. Floor abstention is not a positive learned intervention.

Primary: identified minus marginal FULL paired completion-lower utility,with
nominal3000-locality-bootstrap95%lower endpoint positive. Also require positive
lower effect over the frozen original guard,positive lower utility over floor,
and complete original recording safety. Known means,auxiliary Brier,MSE or
support counts cannot replace these conditions. Cost seeds17/29/43 share
forecaster43;repeated views are not independent recordings or forecast pipelines.

Registered secondary diagnostics: native normalized moment MSE,easy-event
Brier,false unsafe admissions,false safe rejections and known reference support.
These use future labels only after causal choices are fixed and do not change
the primary. Independent selection/calibration/confirmation and reserved
DroneCrowd remain unopened. This is SOURCE fitting/resubstitution only.

## Runtime And Reporting

Native arm64 `.venv-pytorch`,CPU4/inter-op1/workers0,no resource probing or
multiprocessing. Atomic50-update checkpoints retain optimizer,sampler,RNG and
objective identity;heartbeat every50. Explicit unchanged-definition resume,
12hresumable phase boundary,10GiBdisk reserve and1GiBprivate checkpoint cap.
Use the actual pilot's resource estimate to decide local versus CREATE;never
shrink the budget because it is slow. No remote job without checking duplicates.

The data remain oblique detector-silver/local-image obs8/pred12/stride12raw
frames. No independent,SDDt+50,metric/seconds,human-gold,true3D,foundation,
physical-safety or new neural-dynamics claim. No deployment promotion follows
from SOURCE fitting. Publish code/configs/reports/light metrics only,not raw
data,features,caches,weights,videos/images or third-party assets.
