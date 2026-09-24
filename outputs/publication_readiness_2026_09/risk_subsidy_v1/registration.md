# Fixed Query Risk Credit and Denominator Factorial

## Material Passport

Registered 2026-09-24 before these new policy decisions and outcome metrics.
Motivated by an explicitly exposed-source diagnosis, not an independent hypothesis
test: nine repeated zero-CV harms represent four windows and three tracks. None
has exactly constant past velocity; all fail the individual signed-risk rule.
An exact-history-CV veto is therefore not pursued as a direct repair. No new
threshold is fitted to those cases. Every fixed arm is reported, including failures.

Four design-exposed SDD sites, seeds 17/29/43, three frozen forecasting actions.
Obs8/pred12 native annotation steps, stride 12, annotation pixels. Neither this
experiment nor site exclusion creates independent calibration or confirmation.

## Hypothesis and Fixed Arms

Let q_i be the freshly fitted expected net easy harm, r_i the fitted easy-CV
denominator, g_i the existing predicted net gain, and z_i a binary intervention.
All coefficients are past-conditioned stored predictions, never observed costs.
The fixed rho is .02. Each problem is within one recording and query frame.

The original rule is sum(z_i*q_i) <= rho*sum(all r_i). Two distinct mechanisms
can relax individual protection: other agents' predicted negative q_i and
denominator from unselected agents. Compare the complete 2x2:

| Rule | Risk coefficients | Denominator |
|---|---|---|
| net_population, same-rule recomputation | q_i | all forecastable targets |
| net_clipped_population | max(q_i,0) | all forecastable targets |
| net_selected | q_i | selected targets only |
| net_clipped_selected | max(q_i,0) | selected targets only |

The clipping is max(E[net harm|X],0), NOT E[positive harm|X]. Within-agent
uncertainty/benefit cancellation remains. No arm is a zero-harm certificate.
Selected-denominator constraints are implemented as sum(z_i*(risk_i-rho*r_i))
<=0. Their coefficients can be negative; never prune individually infeasible
agents before joint optimization. The canonical feasibility test separately
uses math.fsum on original risk and denominator, followed by multiplication
by the fixed binary64 .02. Small queries (at most ten eligible agents) use
exhaustive optimization of that ORIGINAL inequality; feasible all-supported
solutions with positive gain are also trivially optimal. Larger nontrivial
queries use a transformed MILP and an original-inequality feasibility check.
They remain explicitly canonical-optimum-unverified even when the numerical
solver reports optimal. Failed checks fall back and are separately marked;
an uncertified feasible solution is not the same as a failed empty fallback.
No risk-tolerance relaxation or post-readout solver change is allowed.

Retain old strict, positive-population, signed-pointwise and parent-net-population
policies unchanged. All four factorial arms are recomputed with the same solver
wrapper and stable original-inequality check; the parent-net control discloses
any numerical/order difference from the previous run. No allowance is widened.
The same-query exact intervention count of net_clipped_selected supplies three
additional count-matched versions of the other factorial arms. A nonoptimal
reference fallback to zero is recorded separately from ordinary zero coverage.
Never force a
rejected row merely to fill a count. Report missing counts and solver fallbacks.
No cross-query, cross-recording or cross-site budget transfer is allowed.

The seven new solver arms optimize the identical frozen g_i on the unchanged
causal eligibility pool; the four retained background policies are not all
optimizers of this contract. No learned head/normalizer/feature change, new training, new neural
forecast, intervention quota tuning, outcome-based model selection or threshold
sweep. Causal histories are not modified, and future availability does not filter
the inference population. The full 175,756 registered rows remain included.

## Outcomes and Interpretation

Freeze every decision before opening outcome archives. Report all 33 action/arm
rows, ADE/FDE equal physical-site gain, three seeds, hard, positive-easy, complete,
partial and unknown support, zero-CV harm, intervention counts, tails and worst
site. Use 3,000 paired physical-site bootstrap draws; four exposed sites and
overlapping windows preclude confirmatory/population safety claims.

Primary mechanism contrasts: same-rule recomputation versus retained parent;
each single restriction versus original net
population; both restrictions versus each single restriction; each matched rule
versus net_clipped_selected. Report point estimates and paired intervals without
choosing a deployment winner. The question is whether pooled credits/denominator
explain unsafe intervention and what utility is lost when each is removed.
Any remaining zero-CV harm rejects strict observed protection. Unknown labels
are not safe. Positive-easy degradation is also reported and must not exceed 2%
for any candidate interpretation. No guarantee follows even if no harm is seen.

Independent source admission/calibration is still required. DroneCrowd stays
closed; DUT remains exposed diagnostic; HT21/CroHD remains quarantined. No
external readout, deployment, Stage5C, SMC, metric/seconds, true-3D, foundation or
submission-readiness claim. Exact replay, separate arithmetic and small-query
exhaustive tests validate computation, not the research hypothesis.
