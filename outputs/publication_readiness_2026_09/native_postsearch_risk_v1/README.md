# Does Risk Learning Transfer After Joint Search?

I trained the risk head on combinations that the search actually chose,
rather than on random candidates or the old single-path decisions.
For each seed and outer held source, seven sources fit the models,
six different sources choose the threshold, and the held source is
scored after the decisions freeze. Choices inside the fitting set
are themselves produced out of source. The shared path search is
refit within each fold, so the numbers below are not a replay of
the earlier thirteen-source path-portfolio model.

| Held-source policy | Switches | Original passes / 102 | Total-risk upper bound above 2% | Undefined easy risk |
| --- | ---: | ---: | ---: | ---: |
| Screen-first search without the additional threshold | 3,682 | 1 | 51 | 51 |
| Donor-selected predicted risk margin | 2,854 | 1 | 46 | 56 |
| Donor-selected post-search risk head | 3,280 | 1 | 46 | 56 |
| Retrospective margin rank at identical counts | 3,280 | 1 | 45 | 56 |

At equal per-recording switch counts and action cardinalities, the
learned rank changes conservative utility by **-0.00035 pp**
[-0.00272, +0.00200] relative to the margin rank. The learned policy is
-0.055 pp [-0.294, +0.089] against CV and -9.664 pp
[-21.805, -0.481] against OLS4. The retrospective count match is not
an online policy. Neither fitted threshold family has an all-pass
operating point in any of the 42 validation folds.

On the 31 source-seed cells in which search emits candidates, the
budget-event Brier score is 0.28670 versus 0.25912 for the
fitting-prevalence predictor. Easy-reference prediction improves
(0.25162 versus 0.28838), but that is not sufficient to predict
risk or satisfy the original recording rule. The 11 cells with no
emissions are not part of Brier estimation; all 102 recording-seed
views remain in the operating risk denominators.

**Decision:** Do not use this risk head. Training on the selected
distribution and using separate validation sources was not enough.
The next candidate needs to account for the size and accumulation
of harm and easy-reference error over a recording, not merely the
probability that one selected query appears locally within budget.
This is a testable next hypothesis, not an explanation proven here.

The readout uses exposed Moving A development: 34 recordings,
14 sources, three frozen forecaster seeds, eight observed and
twelve predicted annotation steps in native pixels. The paired
intervals use 3,000 source resamples after seed averaging. No
external outcomes, deployment change, Stage5C or SMC.

[Protocol](protocol.md) | [Metrics](analysis.json) |
[Fit receipt](fit_execution.json) | [Replay receipt](replay_execution.json)

```sh
.venv-pytorch/bin/python scripts/run_m3w_postsearch_risk.py fit
.venv-pytorch/bin/python scripts/run_m3w_postsearch_risk.py score
.venv-pytorch/bin/python scripts/run_m3w_postsearch_risk.py replay
```

Fit checkpoints one held-source and seed combination at a time.
Resume the same phase to reuse completed folds. The private inputs,
choices and model coefficients are not committed.
