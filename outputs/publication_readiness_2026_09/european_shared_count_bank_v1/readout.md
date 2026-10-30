# Reoptimized physical and neural banks at a common query count

## Provenance

The protocol, code and allocation checks were committed before the real pilot and these outcomes.
fresh_run: current causal inference, both bank optimizations, complete-policy scoring and bootstrap.
cached_verified: 18 fixed unit-admission cost heads, frozen history forecasts and source packets.
The old joint choices are hash-verified cached controls, not newly solved again.
No new neural training. All 72 groups and 144 legacy full-policy controls reproduce in a separate full replay.
Readout PID 32631: 724.190s, peak RSS 9810214912B.
Replay PID 34909: 753.267s.
SOURCE fitting/resubstitution only. These are not independent efficacy or deployment results.

## Controlled question

Does a frozen neural forecast add value after the physical-only controller is also optimized?
Every current recording/frame query uses the physical independent policy's intervention count k.
All four controls use exactly k. Zero-count queries keep the unchanged floor in every control.
Six physical actions: constant position, causal finite-difference velocity, damping 0.90 and 0.97,
history OLS4 velocity and history OLS8 velocity. The expanded bank adds the frozen history forecaster.
The trimmed neural independent arm keeps its k largest predicted gains, with canonical row-ID ties.
Both joint optimizations use the same all/easy signed predicted 2% budgets and fixed costs.
The neural joint solver retains the better feasible warm start of optimized physical and trimmed neural.
Predicted bank nesting is checked for every query; it is not an observed-risk guarantee.

## Full-cohort utility and original recording risk

ADE is computed over available future labels. Values are locality-averaged gain over the frozen floor.
95% intervals: 3,000 locality bootstrap draws after context/head-seed averaging over 12 exposed localities.
These conditional exploratory intervals do not adjust for the earlier development search.

|Policy|Known ADE gain % [95% CI]|Supported recording views|Known all-risk violations|Known easy violations|Undefined easy|Selected row views|
|---|---:|---:|---:|---:|---:|---:|
|physical_independent|+1.710155 [+0.611688, +3.313663]|55/444|130|281|43|262284|
|neural_trimmed|+1.900218 [+0.654925, +3.707069]|50/444|150|305|43|262284|
|physical_joint|+2.322960 [+0.785519, +4.512446]|54/444|219|302|43|262284|
|neural_joint|+2.544091 [+0.855495, +4.944945]|47/444|229|322|43|262284|
|legacy_neural_independent|+1.904562 [+0.660504, +3.710170]|46/444|149|308|42|265671|
|legacy_neural_joint|+2.552950 [+0.861795, +4.959240]|43/444|231|324|42|265671|
|floor|+0.000000 [+0.000000, +0.000000]|0/444|0|0|444|0|

Legacy policies retain their original seven-action counts and are contextual controls,
not common-count primary comparators. The no-switch floor's selected-risk ratio is undefined,
not a safety pass. Same-count contrasts establish utility comparisons, not deployment qualification.

|Prespecified contrast|Known increment pp [95% CI]|Wholly-unknown completion-lower pp [95% CI]|
|---|---:|---:|
|neural_joint_minus_physical_joint|+0.221131 [+0.065452, +0.430560]|+0.154613 [+0.005466, +0.362079]|
|neural_trimmed_minus_physical_independent|+0.190063 [+0.043905, +0.383298]|+0.160477 [+0.014986, +0.356533]|
|physical_joint_minus_physical_independent|+0.612804 [+0.161424, +1.214384]|+0.485181 [+0.065016, +1.066250]|
|neural_joint_minus_neural_trimmed|+0.643873 [+0.179558, +1.262745]|+0.502301 [+0.071374, +1.096563]|

The primary is neural_joint minus physical_joint. The four-policy interaction contrast is secondary:
known +0.031068 [-0.003349, +0.071656]; completion-lower -0.012419 [-0.061671, +0.027880].
It asks whether joint allocation changes the incremental value of the neural bank compared with
the independent controls. Its own unknown bound pairs positive/negative forecast terms and takes
the smaller of two valid Lipschitz disagreement bounds, allowing shared terms to cancel.
It is not a subtraction of two lower bounds and not a causal effect on real agent behavior.

## Numerical and safety checks

Repeated query views: 112596; multi-agent: 86316; zero-count: 50924.
Solver reasons: {'physical_joint': {'zero_count_exact': 50924, 'exact_milp': 53094, 'unconstrained_optimum_feasible': 6643, 'singleton_pointwise_exact': 1931, 'postcheck_failed_parent_fallback': 4}, 'neural_joint': {'zero_count_exact': 50924, 'exact_milp': 53318, 'singleton_pointwise_exact': 2017, 'unconstrained_optimum_feasible': 6330, 'postcheck_failed_parent_fallback': 7}}.
Nonoptimal query views: {'physical_joint': 4, 'neural_joint': 7}.
Neural feasible warm starts: {'trimmed_neural_independent': 79481, 'optimized_physical': 33115}.
Overlapping neural-joint recording failure conditions: {'all_selected_risk_upper': 336, 'easy_selected_risk_upper': 387, 'easy_degradation_upper': 272}.
Any solver rejection retains its checked feasible anchor. A nonoptimal fallback is reported,
not renamed an exact optimum. Replay agreement does not turn a failed risk gate into a pass.

|Head seed (same frozen forecaster)|Primary known pp [95% CI]|Primary completion-lower pp [95% CI]|
|---|---:|---:|
|17|+0.222168 [+0.058530, +0.434830]|+0.178356 [+0.016707, +0.394221]|
|29|+0.206316 [+0.060343, +0.382558]|+0.130321 [-0.021514, +0.313246]|
|43|+0.234908 [+0.063350, +0.476435]|+0.155161 [+0.005862, +0.389261]|

|Exposed locality|Primary known pp|Completion-lower pp|Completion-upper pp|
|---|---:|---:|---:|
|eu-locality-007|+0.067252|+0.063664|+0.070840|
|eu-locality-008|+0.404592|+0.233567|+0.575617|
|eu-locality-020|-0.042588|-0.042588|-0.042588|
|eu-locality-048|+0.064152|+0.057536|+0.070768|
|eu-locality-067|+0.108476|-0.189266|+0.406218|
|eu-locality-074|+0.113760|+0.049606|+0.177914|
|eu-locality-082|+0.000000|+0.000000|+0.000000|
|eu-locality-110|+1.153395|+1.119921|+1.186870|
|eu-locality-112|+0.025470|-0.005908|+0.056847|
|eu-locality-119|+0.170870|+0.164896|+0.176844|
|eu-locality-124|+0.004921|+0.003006|+0.006836|
|eu-locality-126|+0.583270|+0.400919|+0.765621|

These locality lower/upper endpoints are missing-outcome sensitivity ranges, not per-locality confidence intervals.
## Fitting gates

- complete72groups: pass
- all144legacy_policy_scores_exact: pass
- same_count_every_current_query: pass
- predicted_bank_dominance: pass
- all_new_queries_solved_optimally: fail
- primary_known_CI_positive: pass
- primary_completion_lower_CI_positive: pass
- complete_primary_recording_risk: fail

6/8 fitting checks; advancement False.
The complete-policy recording criterion remains the original 2% requirement. No safety threshold,
count anchor, label cohort or model is chosen from these readout results.

## Limits

The same 149,510 unique agent-target rows form 18,766 distinct current-frame queries
across 74 recordings and 12 localities. Repeated row views: 897,060.
Three cost-head seeds are not three new forecasters or independent datasets. Query targets are
not guaranteed to cover all visible agents. No pairwise interaction term is optimized.
Wholly-unknown completion bounds do not also bound missing interior labels of partial ADE rows.
Full-roster FDE is not estimable when a required endpoint is unavailable. Per-group FDE values
remain in the aggregate bundle rather than inventing a whole-roster value.
Future labels are scoring-only; inference takes past-only features, fixed forecasts and current query IDs.
Current task is obs8/pred12 at raw stride12, not supplementary raw-frame t+50.
No independent confirmation, deployment change or submission-readiness claim.
Detector-silver, image-local coordinates are not metric, seconds-level, human gold, true3D or foundation evidence.
Stage5C is not executed and SMC is disabled.
