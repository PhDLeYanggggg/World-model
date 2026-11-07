# Most Positive Hindsight Capacity Is Already In The Causal Bank

The same fixed per-query intervention caps and original recording
rule are used for all three banks. The comparator is OLS4 and the
normalizing reference is CV. These are label-aware optima on the
exposed Moving A development set, not forecasts from a deployable
selector.

| Forecast alternatives, in addition to CV fallback | Risk-feasible views | Positive lower utility vs. OLS4 | Unresolved objectives |
|---|---:|---:|---:|
| Five causal alternatives | 96/102 | 48/102 | 0 |
| Same causal bank plus three learned families | 99/102 | 51/102 | 0 |
| Same native bank plus the prior shared-motion transform | 99/102 | 51/102 | 0 |

The 102 views repeat 34 recordings in 14 source groups across the
same three forecaster seeds. The five causal alternatives are
constant position, two damped-velocity variants, OLS4 and OLS8.
The learned alternatives are the fixed-head EqMotion adaptation,
Past Context Transformer and self-kinematic forecast.

**Three newly positive views represent one recording in one source.**
That recording was already risk-feasible with the causal bank, but
its exact conservative OLS4 gain was -0.1582 pp in each seed.
With the learned alternatives it is +5.0501, +4.3426 and +4.2842 pp.
The other three-view improvement, from no feasible risk solution
to feasibility, occurs in a different recording. It does not
establish positive OLS4 utility there.

On 96 views where both bank optima exist, adding learned alternatives
raises median optimal lower utility by 1.382 pp. The source-balanced
positive-rate gain is +7.143 pp with a 3,000-draw paired source
bootstrap interval of [0.000, 21.429] pp. This rate gives each
source equal weight after seed averaging, so it is not the raw
three-view difference divided by 102. Adding the shared-motion
option changes median optimum by +0.006 pp on the 99 comparable
views and adds no positive view.

**Implication for the paper:** The additional learned paths can
matter, but most positive capacity is already present in this
causal menu. A future learned selector has to beat a matched
causal-menu selector, not merely show headroom over a single CV
or OLS4 forecast. The finding does not establish that a past-only
rule can find the one newly positive case on an unseen recording.

## Reproduction

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_neural_bank_capacity.py score
.venv-pytorch/bin/python scripts/audit_m3w_neural_bank_capacity.py replay
~~~

The original registered implementation stopped after 14 cells because
it used an upper bound from a restricted recipient set in a larger
set. The revised registration fixes only this audit: a checked
restricted witness may carry over, but its upper bound may not.
The original checkpoints were not deleted. The revised run solved
204 new bank/view cases, including three timed refinements, in
161.3 seconds on local CPU. It reconstructed and exactly verified
the existing expanded-bank summary on the same roster. The 1.5-second
aggregate replay is cached verification, not a second solver run.
Eight targeted tests passed.

[Protocol](protocol.md) |
[Revised registration](registration_v2.json) |
[Metrics](analysis.json) |
[Score receipt](score_execution.json) |
[Replay receipt](replay_execution.json)

No new training, external data access, independent result, deployment,
Stage5C or SMC. Results use eight observed and twelve forecast native
annotation steps and image pixels, not seconds or metric geometry.
