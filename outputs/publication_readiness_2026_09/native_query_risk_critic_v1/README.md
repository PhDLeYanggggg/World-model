# Native Query Switch-Set Risk Critic

**Decision: not admitted for external confirmation.** At almost the same
switch count, predicting risks for a proposed set improves a local diagnostic
but does not improve the registered support prediction or satisfy the
original 2% recording rule.

The [registered protocol](protocol.md) fixes the native 8/12 bank, scoring,
and CV and OLS4 references. Three seeds and 14 outer source exclusions give
42 fits per critic arm, with a second source exclusion when constructing
training proposals. The 34 recordings are exposed development data, not
independent confirmation. The analysis uses 3,000 paired source bootstrap
draws after seed averaging, not a row bootstrap. Coordinates are pixels,
and the horizon is in annotation steps, not calibrated seconds.

| Screen | Accepted sets | Locally feasible sets accepted | Agent switches | Original rule passes / 102 | Undefined easy risk | All-risk upper >2% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old gate | 45 | 10 | 90 | 3 | 84 | 15 |
| Recalibrated sum | 13 | 3 | 26 | 0 | 98 | 5 |
| Query critic | 44 | 21 | 88 | 4 | 89 | 9 |

On all 4,839 identical seed-query proposals, easy-support Brier score
is 0.14933 for the critic vs. 0.14724 for the recalibrated sum. The paired
reduction is -0.00209 (95% interval [-0.01227, +0.00563]). None of the
other three normalized absolute prediction-error reductions have intervals
strictly above zero. Locally feasible means positive completion-lower gain,
both original excesses nonpositive and a positive easy reference for the
proposal itself. The selection counts are not exactly matched, and this
local enrichment is not the recording risk outcome.

The critic's completion-lower utility is +0.00411 pp [0.00026, 0.00864]
versus CV, -0.00117 pp [-0.00859, +0.00426] versus the old gate and
-9.820 pp [-21.997, -0.552] versus OLS4, with a common CV denominator.
A negative lower bound does not prove that the true effect is negative.
The thresholds were not adjusted after scoring.

## Proposal Capacity

The posthoc [label-aware audit](proposal_capacity.json) separates structural
support from scoring failures. Every recording has positive easy reference
in its full roster (31,520 easy rows total). In **24/102** recording-seed
views, no fixed joint proposal contains any such row. No policy that only
accepts or rejects those proposals can pass the original support rule on
those views. In **57/102** views, accepting one particular fixed query and
retaining CV elsewhere passes the original scorer at positive local lower
gain. Those are hindsight witnesses, not a learned policy or an exact global
ceiling. In the critic arm, 87 views make no switches and two more switch
without reaching an easy reference.

The next model change should therefore affect *which sets are proposed*.
Compare a support-aware shortlist with the current gain-oriented shortlist
at the same candidate count and keep the final gate identical. This tests
retrieval before adding another gate. The present ridge screen is empirical,
not a conformal guarantee: [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
assumes a bounded-above, monotone loss and exchangeability in its basic
result; the selected-reference ratio does not inherit that guarantee.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_query_critic.py fit
.venv-pytorch/bin/python scripts/run_m3w_native_query_critic.py score
.venv-pytorch/bin/python scripts/run_m3w_native_query_critic.py replay
.venv-pytorch/bin/python scripts/audit_m3w_native_query_critic_capacity.py
```

[Fit receipt](fit_execution.json), [metrics](analysis.json),
[score receipt](score_execution.json), [replay receipt](replay_execution.json).
The prior gate is reproduced exactly. No forecast is retrained; the fresh
fits are the decision critics. No raw data, decisions or model arrays are
committed. No identity exception, external outcome access, deployment
change, Stage5C execution or SMC.
