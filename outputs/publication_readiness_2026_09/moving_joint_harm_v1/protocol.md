# Joint Event Supervision Under Unchanged Risk Rules

Prospective exposed-SOURCE development experiment. This is not an independent
test, a new forecasting model, or a claim that hurdle factorization is novel.
The user retained the original rules: no identity certification, no empty
control counted as success, and no change to 2% risk or support requirements.

## Hypothesis And Prior Negative Evidence

Source-disjoint point critics still underestimate selected H and EH. Their
veto reduces less harm than the benefit it loses. EuropeanSquares hurdle and
ranking studies already found mixed occurrence gains without reliable matched
selection gains; Moving's easy-reference hurdle also failed to beat direct
regression. This test is narrower: jointly supervise exclusive benefit,
non-easy harm, easy harm and zero events for the same seven-amplitude query
model, including coherent conditional magnitudes. It may fail again.

## Controlled Change

Two equal-architecture arms: product_moment and event_supervised. Both use
width64, identical initialized weights, exact same source-balanced query draws,
own-part preprocessing, AdamW settings, 2000 updates and the original
bound-aware five-moment MSE plus 0.5 net-gain MSE. Event_supervised adds unit-
weight unweighted categorical cross entropy and positive-event conditional
envelope-fraction MSE. These two auxiliary components form one factorization
supervision treatment; this does not identify their separate causal effects.
No coefficient search, new feature, forecast, threshold or official role.

The head outputs four exclusive event probabilities and three conditional
fractions. B=min(envelope*p_b*q_b,R), H=envelope*(p_h*q_h+p_eh*q_eh),
EH=envelope*p_eh*q_eh. Thus B+H<=envelope and EH<=H. R/ER retain the original
reference branch. Zero is exact zero under the original available-label
definition, not a new tolerance. Future labels are loss/evaluation only.
Event CE uses only observed available-label futures. Conditional MSE uses
only observed positive events; unsupported minibatches contribute no invented
negative event. Unknown futures keep the original conservative bound loss,
never positive/negative occurrence labels. Partial labels retain their original
available-label meaning; no complete future is imputed.

Query averaging is retained. Event CE averages known legal actions per row,
rows per supported query, then supported queries. Conditional MSE does the
same on positive-event actions. A source with no fitting positive event is not
claimed to have conditional-magnitude support. Fitting-only query/source
balanced categorical priors provide event-score controls; the unsupervised
product factors alone are not calibrated probability comparators.

## Roles And Fixed Budget

Reuse all18 producer/critic contexts and the original three-way inner-C map,
three producer seeds17/29/43, all54 own-part fits per arm:108new heads and
216000 primary updates. All new models freeze before any new nested-C score.
Six real50->100/resumed-vs100/direct cells test numerical stability/resume
before full fitting. The first full model is independently refitted exactly.
CPU4/interop1/workers0 native arm64; checkpoints and heartbeat every100updates.
5GiB disk reserve; no HPC submission unless actual resource needs justify it.

For each heldC part, fixed parent point proposer from A is risk-vetoed by
each new critic fitted only on B; swap A/B. All preprocessing, priors and
labels exclude the held part. Proposer remains exactly the frozen parent,
not a newly chosen best model. There are108 orientation blocks,276 required
source views and696 recording views. Repeated windows are not independent.
No new H readout. All23 supported source units remain exposed development;
original24-group primary remains not_estimable, not passed. Reserved roles
stay closed. No deployment, Stage5C or SMC execution.

## Scoring And Decision

Retain original risk-only veto: the fixed proposal can only be kept or rejected
using predicted H<=.02R and EH<=.02ER with positive references. The veto cannot
reselect an amplitude. Within every query, each arm has its own causal
utility-ranked control with exactly the same count of every amplitude.

Primary: event-veto minus its own count/amplitude-matched control, direct
completion-lower contrast on causal nonempty queries. Any required empty view
makes the complete primary NA, not zero, and no intervention is forced.
Secondary: full-strategy matched contrasts and event-veto minus product-veto;
the latter has differing intervention rates and is not a ranking-only claim.
Keep all full-policy original recording risk/easy-net/support checks. No
primary-to-secondary substitution. Pass requires a positive primary lower CI
and every original full-policy risk/support view passed. Training or prediction
improvements alone cannot pass this decision.

Mechanism diagnostics: on all legal known actions and the exact same frozen
proposer-selected known actions, query-balanced H/EH envelope-fraction MSE,
R normalized MSE, H/EH Brier/log loss and fitting-prior Brier. Report paired
product-minus-event error reductions, not just ratios after changed selection.
Selected known cost masses remain descriptive checks, not risk certificates.
Intervals use3000 source-group resamples after context/seed/orientation
averaging, conditional on these23 exposed source units; no IID-scene theorem.

## Reproduction And Limits

Run register, pilot, train, evaluate, verify with
`.venv-pytorch/bin/python scripts/run_m3w_moving_joint_harm.py PHASE`.
Resume an interrupted phase with `--resume`; completed receipts are immutable.
Code/config/protocol/registration must be committed before fitting. New data
arrays and weights stay local and ignored; public outputs contain aggregate
metrics and hashes only. Unit checks, real training, replay and scientific
success are reported separately. Annotation-order obs8/pred12 image pixels,
not t50, seconds, metric, human gold, true3D or foundation evidence.
