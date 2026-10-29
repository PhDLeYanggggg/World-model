# Fixed-Neighbor Local Cost Objective

## Material Passport

Registered before new real-data fitting. This is an exposed-TRAIN diagnostic,
not an independent evaluation or a new neural forecaster. Follow-up to the
completed policy-conditioned analogue experiment; its magnitude gate failed.
Keep obs8/pred12 rawstride12, pixel/local detector-silver, the 2% positive
selected harm/reference budget, and every reserved data role unchanged.

## Hypothesis and Estimator

Uniform averaging of H/envelope minimizes fraction error, not necessarily the
source-normalized signed-risk error used for evaluation. Test a local quadratic
objective with the same 32 neighbors, frozen selected query mask, donor recording
exclusion, features and full-bank normalization. No new k, threshold or feature
search. No query future outcomes, knownness or test goals enter prediction.

For each target (all harm H with reference R; easy harm EH with reference ER),
let E be the causal envelope, S the fixed TRAIN source scale and Rhat the frozen
original reference prediction. Donor-only loss for a constant local fraction f:

    sum_i [(E_i*f - .02*Rhat_i) - (H_i - .02*R_i)]^2 / (N_source(i)*S_i^2)

Here N_source counts selected, known donors of that source AFTER removing the
query's whole source-qualified recording, using the full reference bank, not the
query labels. This gives equal-source weighting in the reference objective; it
does not prove unbiased estimation of query risk or equal independent evidence.
The bounded closed-form solution is clip(sum(w_i*t_i)/sum(w_i),0,1), where
w_i=E_i^2/(N_source(i)*S_i^2) and t_i=(H_i+.02*(Rhat_i-R_i))/E_i.
Output E_query*f, copying original B/R/ER. Count rather than hide B+H envelope
or EH>H violations; no posthoc feasibility projection changes the comparison.

This changes the regression objective, including envelope, source-scale/source
frequency weighting and frozen reference offsets. It does NOT isolate these
individual weights. Fixed secondary controls identify reference-offset effects
and whether local matching adds information:

1. Frozen original head.
2. Selected-nearest32 uniform fraction mean, exact previous numerical control.
3. Weighted_harm32: same weights, no reference offset.
4. Local_signed32: PRIMARY, complete objective above.
5. Global_signed_mean: same objective on all legal selected donors, no neighbors.
6. Shuffled_signed32: paired corrected fractions permuted within donor recording
   using seed20261029; original causal weights and neighbors retained.

The signed target may lie outside [0,1]; constrain the optimum, not its labels.
An analytic donor-loss check compares the optimum with the prior uniform fit.
This check is true by optimization, not evidence of prediction/generalization.
Report effective donor count and maximum normalized weight to expose leverage.
Unknown queries remain; insufficient donors yield undefined, never safe zero.

## Frozen Evaluation and Stop Rule

All six contexts x three existing head seeds x four sources = 72 views;
same 62,902 repeated queries. No new forecaster seeds. Equal-view primary MSE,
not ADE/FDE, is unchanged. The five fitting conditions now compare against the
improved selected-neighbor control, not the weaker full-bank predecessor:

- Every view has full donor support.
- Primary easy MSE beats the original head.
- Primary easy MSE beats selected-nearest32.
- Primary all MSE is no worse than selected-nearest32.
- At least 12/18 context/head-seed pairs beat both easy-MSE controls.

Strict differences use 1e-12 slack as before. This is a tighter continuation
screen, not a relaxation. Even a pass does not authorize deployment or opening
independent roles. A failure leads to a recorded mechanism diagnosis, not
posthoc primary-arm substitution, source deletion or threshold search.
Also report source/context/seed slices, observed versus predicted harm,
event/non-event harm SSE, unsupported counts and moment violations.

## Execution and Boundaries

Commit protocol, code, tests and immutable registration before the real pilot.
Pilot tests unchanged predictions/neighbors/normalization and runtime only;
then run all 72 views and exact inference/retrieval replay. Native arm64 CPU4,
interop1, workers0, numerical threads4 and nearest jobs1; use current authorized
read-only CREATE TRAIN packets. No new scheduler job or private upload. Atomic
receipts, process lock, heartbeat, 10GiB reserve, 12h boundary and explicit
resume protect work. No full-cache materialization or expensive neural refit.

Donor labels supervise fitting; query labels score only. Inner-held is discarded
before forward; outer development and independent roles remain unopened.
Upstream exposure prevents a whole-pipeline OOF claim. No confirmatory CI or
p-value from these fitting repeats; no simulation/metric/seconds/human-gold/
physical-safety/true3D/foundation/submission-ready claim. Stage5C/SMC stay off.
