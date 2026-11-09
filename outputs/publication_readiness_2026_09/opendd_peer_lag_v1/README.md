# Does a neighbor help after it predicts two held observed steps?

The [previous comparison](../opendd_peer_transient_v1/README.md) showed
that a nearby agent moving in the same direction is not, by itself, a
reliable source of forecast improvement. I tested a more specific
rule: a neighbor must predict the target's change in velocity better
than both the target's own lagged motion and a zero-change reference
on the last two **observed** steps. The neighbor and regression are
chosen before those two steps. Both forecast arms use the same gate;
only the source of the lagged change differs. See the [protocol](protocol.md).

The candidate banks were frozen in `e1f906d3a` before scoring. Eight
observations, twelve forecast steps, the CV and OLS4 references, the
additive reference, all single-edit choices and the three original
2% risk checks are unchanged. Each menu has 3,690 edits at the same
query-specific peak slots. The three seed budgets share the same 202
development recordings in seven locations.

## Support shrinks before forecast labels are used

Of 4,015 agent-window rows with a valid prefix neighbor, 769 (19.2%)
pass the two-step prediction check. These row counts are not multiplied
by the three seed budgets and are not independent observations.
After matching the original budgets, each arm adds 764 row-slot options
over 122 of the 439 edited recording-seed views. Direct replacement
at the original additive recipients changes 99 paths in each arm;
all 99 paired paths differ across arms. The replacements occur at
`rdb1` (28), `rdb5` (3), and `rdb6` (68), across all three budgets.

## The added lag does not expand the useful menu

| Menu | Full-grid positive without risk or accuracy constraints | Full-grid positive under all constraints | Distinct recordings under all constraints |
| --- | ---: | ---: | ---: |
| Original expanded menu | 233 / 439 | 21 / 439 | 13 |
| Plus own lag | 233 / 439 | 21 / 439 | 13 |
| Plus neighbor lag | 233 / 439 | 21 / 439 | 13 |

All solves returned optimal or infeasible. The constrained score
requires positive conservative accuracy versus CV and OLS4, no loss
versus the common additive reference, original risk compliance, and a
positive full-grid pair-error gain. These counts are label-aware
possibilities, not the causal rule's results.

| Direct neighbor lag minus own lag (positive is better) | Location mean | 95% location bootstrap |
| --- | ---: | ---: |
| Conservative individual gain (pp of CV error) | -0.0000313 | [-0.001444, +0.001819] |
| Co-observed pair gain (native units) | -0.00000837 | [-0.00001732, -0.00000033] |
| Full-grid pair lower gain (native units) | -0.00001285 | [-0.00002595, -0.00000218] |

Both causal replacement arms qualify for risk and individual accuracy
on 34/439 views versus the original additive reference, and both have
only three views that also improve relative motion. Directly compared
with the own-lag arm, zero views satisfy all constraints and improve
relative motion. The co-observed pair metric is negative at all three
locations with replacements, so the missing-step debit alone does not
explain the result. The magnitude is small; the interval is descriptive
and comes from six estimable development locations with overlapping
training folds, not independent confirmation.

**Decision:** The two-step prefix test found local predictive
correlation, but it did not select edits that transfer to the twelve-step
forecast. I would stop tuning this interaction feature on these locations.
The paper can use this as a matched negative result: short observed
predictability does not guarantee a useful joint forecast under the
original budget and risk constraints. The next priority is evidence on
source-separated, adequately labelled interactions, not a larger pair
head. This is not a test of driver causation or calibrated car-following
parameters; those require different evidence (see [Kesting and
Treiber's trajectory-based calibration study](https://arxiv.org/abs/0803.4063)).

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_peer_lag.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_peer_lag.py score
.venv-pytorch/bin/python -c 'from scripts import run_m3w_opendd_peer_lag as r; print(r.paired_interval(r.study.read(r.OUT / "readout.json")["sites"], "pair_native", 0))'
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_peer_lag.py tests/test_m3w_peer_transient.py tests/test_m3w_interaction_frontier.py
```

`fresh_run`: candidate generation and both comparisons. `cached_verified`:
existing source histories, held-source neural menu, edit budgets and
unchanged score receipts. Native arm64 CPU, four threads, no worker
processes, and no new neural fit. Each fold writes a hash-bound checkpoint
and PID heartbeat; repeat the same command to resume. No confirmation
outcomes opened, no deployment, Stage5C or SMC, and no metric or seconds
claim. [Readout](readout.json) | [Freeze](freeze.json).
