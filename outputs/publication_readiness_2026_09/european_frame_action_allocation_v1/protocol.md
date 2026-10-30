# Fixed current-query multi-action allocation experiment

## Material Passport

Mode:real-data code experiment; previous goal turn:progress. The preceding
complete norm-ratio repair loses utility and fails risk. This experiment tests
decision coupling without a new predictor, loss-weight sweep or test selection.
All results are exposed SOURCE fitting/resubstitution, not independent or whole-pipeline OOF.

## Difference from existing experiments

Earlier binary easy-allocation experiments had one proposal and geometry terms;
the exact learned record policy aggregated full recordings. Neither implements
the present seven-action bank with the current detector-history producer,
current fitted cost predictions and contemporaneous matched-count allocation.
This experiment is budget coupling, not learned pair interaction or new dynamics.
It uses all current query targets; these are not guaranteed to include every
visible agent. No future window is used to define a query group.

## Fixed predictors, inputs and population

Reuse all18 frozen unit-admission cost heads:six contexts,seeds17/29/43.
The six physical candidates and current frozen neural candidate, normalizers,
causal availability, query IDs and original allowed fitting sources stay fixed.
Recompute the same past-only predictions. Require their hashes, independent
choices and complete policy scores to reproduce the earlier readout exactly.
No new supervised training is claimed. Reserved roles remain closed.

Grouping key is source/recording/current raw frame, canonicalized by row ID.
Rows with unknown future outcomes remain in every inference decision. No
labels, future support masks, eventual endpoints or oracle choices enter the
allocation API. Goals, velocities and normalizers retain their verified causal
or fitting-only origin. No central velocity or test endpoint goals are added.

## Four fixed decision arms and floor

- Independent:the original pointwise positive-gain and all/easy2% ratio decoder.
- Frame uniform:one common action for every query target, if causally allowed,
  individually positive predicted gain and jointly within both budgets; otherwise floor.
- Joint matched:choose at most one action per target, with exactly the original
  independent intervention count in EACH current query, maximizing predicted gain.
- Joint free:same allocation without the matched count, as a rate-changing secondary.
- Floor:the unchanged no-intervention reference, not an artificial selected action.

Eligible actions retain positive predicted net gain and positive predicted
all/easy reference. Pooling moves only the two risk inequalities to query level:
`sum(H-.02 R)<=0` and `sum(H_easy-.02 R_easy)<=0`. Every query obeying these
inequalities implies the corresponding predicted recording inequalities, but
does NOT establish observed risk, calibration or easy support. No observed
benefit is used to cancel positive observed harm in the evaluation contract.

Solve the multiple-choice binary problem with SciPy/HiGHS, exact relative gap0,
2s per query. An unconstrained rowwise optimum is certified directly if it
satisfies both budgets; zero-count and singleton queries have exact shortcuts.
Normalize constraint rows separately to preserve tiny easy-channel coefficients.
Postcheck raw signed budgets, integer assignments, gain and per-query count.
Timeout/nonoptimal/numerically invalid output falls back to the frozen independent
decision, not a newly tuned policy. Report all such cases; no exact-solver claim
for fallback cases. Save recoverable private action progress every250 queries.

## Primary evaluation and diagnostic

Primary:joint-matched minus independent full-policy known ADE utility and
wholly-unknown completion-lower utility. Compare uniform and free arms
secondarily. Original all/easy2% complete-policy recording requirements and
easy-degradation condition remain; undefined support is not a safe zero.
Count-matched trimming contrasts remain utility-only, with empty contrasts
not estimable. FDE stays undefined when required final endpoints are missing.
Use all72 groups and3,000 locality bootstrap draws after context/seed averaging.

Recompute the same label-aware whole-record bank oracle and require equality
with the cached148 deduplicated record/context opportunities. Check its actual
allocation against every current frame's signed budgets, recording defined
all/easy support separately. This is a stronger grouping check on an already
label-aware witness, not a deployable controller or a matched-rate optimum.
Do not infer causal identifiability or complete-frame easy support from it.

Record pair-distance contraction relative to floor among the current query
targets using predicted paths in a common coordinate frame. This is a
descriptive dimensionless geometry proxy, not a training objective, all-visible
interaction measure or physical collision/safety certificate.

## Runtime, freeze and reporting

Before full scoring, run the first context/source/seed17 on the first64 and
largest64 causal query groups, deduplicated, with no outcome scoring. Include
a real interrupted action-checkpoint recovery and direct repeat. Pilot-based
runtime is an estimate, not a full result. Run the complete fixed roster even
if the pilot is slow, unless a real resource/runtime hard boundary intervenes.

Native arm64 CPU4/interop1/workers0,12h bound,10GiB disk reserve,256MiB private
progress cap,atomic saves,heartbeat and explicit resume. Local execution is
appropriate unless the real pilot indicates otherwise. No duplicate HPC jobs.
Freeze code/config/protocol before the pilot. Require separate complete
inference/decision/metric replay. Code,aggregates,reports and figures are public;
raw data,features,individual actions,weights and progress caches remain private.

The task is8 observed/12 requested steps at raw stride12, not supplemental
rawt+50. No metric,seconds-level,human-gold,true3D,foundation,Stage5C or SMC claim.
Even positive fitting gates do not establish independent or submission readiness.

## Prior-work boundary

[JFP](https://proceedings.mlr.press/v205/luo23a.html) already models interactions
in a structured multi-agent output distribution. This budget-only allocation
is not a reproduction of JFP or a claim that joint prediction is new.
The solver uses the documented integer/bound/linear-constraint interface of
[SciPy milp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).
Optimization itself is not a method contribution; useful, calibrated decisions
and independent evidence would still have to be demonstrated.
