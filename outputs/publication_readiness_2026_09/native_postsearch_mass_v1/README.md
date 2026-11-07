# Does Fitting Full Recording Masses Improve Risk?

I replaced the post-search probability head with continuous estimates
of selected positive harm, unknown-path charge and observed reference.
Both arms share the same search choices and causal, chronologically
ordered ledger. The only fitting difference is whether the loss
includes the exact total error on each fitting recording. Both use
seven fitting, six donor and one held source per fold, with inner
source-excluded search decisions. The unfiltered search control
replays the prior risk summary exactly.

| Held-source policy | Switches | Original passes / 102 | All-risk violations | Undefined easy risk |
| --- | ---: | ---: | ---: | ---: |
| Same screen-first search | 3,682 | 1 | 51 | 51 |
| Pointwise mass fit, causal ledger | 388 | 4 | 19 | 78 |
| Recording mass fit, causal ledger | 280 | 3 | 16 | 82 |
| Pointwise rank at matched counts | 280 | 4 | 15 | 82 |

At 280 matched switches, the recording-mass arm changes conservative
ADE utility by **-0.000023 pp** [-0.000223, +0.000137] and passes one
fewer view. This matched control is retrospective, not an online
policy. The causal pointwise ledger gains +0.00472 pp
[+0.00032, +0.01145] over CV, but is -9.820 pp
[-21.997, -0.554] versus OLS4. It abstains much more often than the
screen-first policy; the table is not an equal-count comparison for
those two arms. Neither trained ledger finds a donor threshold that
passes all required recordings in any of the 42 folds.

On the same nonempty held search outputs, full-recording fitting
worsens normalized all-excess MSE from 0.0351575 to 0.0357293
(change +0.0005718 [+0.0000152, +0.0014161]).
The easy-excess change is +0.0003687 [-0.0001048, +0.0012507].
These errors cover 66 nonempty recording-seed views in 12 sources;
all 102 views remain in the operating evaluation. The interval
resamples sources after seed averaging, not individual queries.

**Decision:** Keep the frozen policy. The original screen's excess
risk is not repaired by this recording loss, and zero expected credit
or nonpositive predicted balances are not actual risk guarantees.
The [separate feasibility case](../native_easy_support_certificate_v1/README.md)
shows why one recording cannot pass with any subset of the present
distinct actions. Any next forecast change must be compared against
CV and OLS4 as well as the original risk requirements.

All 42 folds are exposed Moving A development, on 34 recordings
from 14 sources repeated over three forecaster seeds. Evaluation uses
eight observed/twelve predicted annotation steps in image pixels,
the original 2% and support rules, and 3,000 paired source draws.
No external data, deployment, Stage5C or SMC.

[Protocol](protocol.md) | [Aggregate scores](analysis.json) |
[Fit receipt](fit_execution.json) | [Replay receipt](replay_execution.json)

```sh
.venv-pytorch/bin/python scripts/run_m3w_postsearch_mass.py fit
.venv-pytorch/bin/python scripts/run_m3w_postsearch_mass.py score
.venv-pytorch/bin/python scripts/run_m3w_postsearch_mass.py replay
```

The fit checkpoints each outer fold. Score and replay use the frozen
choices and recheck the prior search comparator. Private weights and
forecasts stay local.
