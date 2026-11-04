# Why Stable Decoder Supervision Did Not Fix Admission

## Evidence And Controlled Change

Fresh_run: 36 real Torch continuations, 72,000 new primary updates, two full
first-job refits, 216 fitting-region views and complete policy evaluation.
Cached_verified: the parent forecast bank and warm states, followed by exact
new cost-output and policy-scalar replay. This is exposed-source development,
not independent confirmation. The original 24-group primary is not estimable;
all 23 supported groups remain in the secondary analysis.

Each of 18 frozen local-history heads receives two equal-budget continuations.
Weights, optimizer state, sampling stream, normalization, inputs and additional
2,000-update budgets are matched. Only the fixed unit logit auxiliary changes.
This changes loss geometry and weighting, not just arithmetic precision. The
trajectory forecaster itself is unchanged.

## Failure Taxonomy

### 1. Precision Was Not The Main Failure

The preceding fitting-only diagnostic found no float32/float64 choice changes.
Float64 recovered less than 1% of selected known easy-harm mass from underflow.
The new auxiliary therefore tests a training intervention, not a proven
floating-point root cause. Attenuated logit sensitivity was real, but does not
establish that decoder saturation alone caused the policy failure.

### 2. Reliability Depends On Which Region The Model Selects

On the frozen parent's selection region, the known easy-harm/predicted
easy-harm mass ratio changes from 653.07 for the parent to 498.10 after ordinary
continuation and 0.485 after auxiliary training. This is a substantial repair
on that fixed old region, not a guarantee on every possible selected set.

On the auxiliary model's own causal selection region, its ratio is still
19.83; its known all-harm/predicted all-harm ratio is 13.05. Pooled known
all/easy positive-harm ratios there are 5.09%/16.81%, already above 2% on
critic-fitting sources. The check was executed and replayed before the new
held readout. Pure held-domain shift cannot fully explain this fitting-side
failure. The result does not uniquely distinguish missing information,
conditional stochasticity, model capacity and objective mismatch.

### 3. Conservatism Lost Utility Without Establishing Safety

The joint-free group-macro switch rate drops from 70.00% for ordinary
continuation to 11.86% for the auxiliary. Known gain over the same frozen
floor falls from 11.39% to 2.08%. The fixed auxiliary-minus-continuation
contrast is -9.31 pp, grouped 95% CI [-15.59,-3.44]. The fixed paired
completion-lower contrast is -16.86 pp [-27.30,-7.69]. These are development
secondary contrasts with 3,000 source-group resamples, not independent final
tests or IID-window intervals.

Only 4/348 repeated recording views satisfy the auxiliary joint policy's
complete original risk/support contract, versus 6/348 for ordinary
continuation. There are 236 known all-risk and 192 known easy-risk violations;
145 views have undefined completion-easy support. Worst observed easy-net
degradation is 227.84%. Fewer positive-harm violations are not a complete pass.
Repeated views share sources and trained producers; they are not 348 places.

### 4. More Training Alone Was Not Enough Either

The parent and ordinary continuation have known joint-free floor gains of
11.33% and 11.39%, respectively. Neither passes the full contract. That small
point difference is not assigned an uncomputed significance claim. All
training budgets finish, all parent/receipt/checkpoint hashes match, and both
complete first-job refits are exact. The failure is not an interrupted run.

### 5. Empty Matched Controls Cannot Rescue The Result

The auxiliary has 16/138 group views with no nonempty same-count control,
compared with 3/138 for ordinary continuation. Full-scope matched utility is
therefore not estimable in either arm. Empty controls are not zero utility
and do not certify safety. Count matching remains a utility comparison only;
the complete deployment policy still faces the unchanged full 2% contract.

## Gate Record

| Check | Outcome |
|---|---|
| Definition committed before continuation updates | pass |
| All 36 real fits and 18 matched pairs complete | pass |
| Both full first-job refits exact | pass |
| All new cost predictions and policy scalars replay | pass |
| Original 24-group primary | not estimable, not passed |
| Fixed 23-group completion-lower contrast | negative |
| Complete original policy risk/support contract | fail |
| Full-scope same-count utility | not estimable |
| New independent calibration/confirmation | not_run; roles remain closed |
| Deployment promotion | no |
| Stage5C execution / SMC | false / false |

Solver bindings and feasibility are verified using saved allocation receipts.
This does not claim bitwise reoptimization of the mixed-integer solver.

## Consequence And Next Decision

Do not deploy this auxiliary or choose a different coefficient, seed or policy
from the exposed held scores. Do not claim that a better old-region fit proves
a safe new-region policy. The fixed-region control rules out that shortcut.
Earlier static selected-region weighting, hurdle and log-loss failures remain
negative; this result is not permission to repeat them without a new testable
mechanism.

The next method experiment needs to test whether selected-region reliability
can be learned without merely rejecting most useful forecasts. A prospective
comparison should isolate one change, retain matched predictor/action banks,
and assess prediction reliability on regions selected by a separate fitted
rule. Independent selection, risk-calibration and final-confirmation sources
remain necessary; reallocating the 20 unused Moving source groups is still
awaiting the requested scientific role decision. No outcome from those groups
was used here. Existing DUT/DroneCrowd roles and the full 2% contract are not
changed to compensate for this negative result.

This does not establish safe M3W admission, a new world-dynamics architecture,
JEPA lift, physical safety, true 3D, foundation status or submission readiness.
Coordinates remain pixels and horizons native annotation steps, not verified
metres or seconds. Stage5C and SMC remain disabled.
