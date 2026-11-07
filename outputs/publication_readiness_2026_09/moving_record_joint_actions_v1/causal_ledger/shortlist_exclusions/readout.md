# What The Portfolio Dropped, And What Six Plus Two Restores

This replay keeps the original forecast bank, direct decisions,
eight-slot budget, source roles, and 2% risk/support requirements.
It uses the exposed Moving A development cohort, with eight observed
and twelve future annotation steps in native image coordinates.

| Retrieval rule | Useful early action in 18 zero-direct views | Useful early action before first direct switch, all 102 views |
|---|---:|---:|
| Original family-diverse ordering | 6 | 9 |
| Family ordering, prefer unused agents | 2 | 8 |
| Cost portfolio without agent preference | 3 | 13 |
| Previous cost portfolio, prefer unused agents | 3 | 12 |
| First six family slots plus two portfolio fills | 4 | 8 |

The four-way replay reproduces the original five losses. In all five,
changing the ranking without agent preference is sufficient to lose
the view's last useful early option. In four of those, adding agent
preference to the family ordering also suffices. One changes only
under the cost ranking among the two individual changes. The original
family list supplied five omitted useful actions in these five views:
three damped-velocity-0.97 forecasts and two self-trained forecasts.
None of the five lost actions' families appears in the corresponding
portfolio query; three agents are still represented by a different
action. This is an action-retrieval diagnosis, not an inference about
pedestrian behavior or a general argument against agent diversity.

Cost rankings retain useful actions in more views across the whole
pre-direct prefix cohort (13 or 12 versus 9), despite losing coverage
in the 18 zero-direct views. The latter stratum depends on the
recording's realized sequence of direct decisions and is not a
feature available at the opening. These descriptive contrasts do not
license routing on that stratum or choosing a test-dependent rule.

## Six-Plus-Two Policy Replay

The fixed hybrid restores three of the five lost views, retains four
of the original six, and adds none beyond the original family list.
Neither the original nor hybrid risk-aware winner picks a positive
supported action in the first two queries of the 18 zero-direct views.
The head is frozen; this does not test a refit on hybrid candidates.

| Calibrated policy | Switches | Complete passes / 102 | Signed easy >2% | Easy upper undefined |
|---|---:|---:|---:|---:|
| Direct only | 6,952 | 2 | 8 | 35 |
| Family-diverse selector | 6,955 | 2 | 8 | 35 |
| Six-plus-two with the same head | 6,952 | 2 | 8 | 35 |

The hybrid adds no opening after calibration. Its paired effect is
+0.00060 pp in completion-lower utility (95% source interval
[0, +0.00180]), compared with the family-diverse selector. At the
same 6,998 switches on identical queries, the effect is -0.00206 pp
[-0.00473, +0.00023] and complete passes fall from 5 to 2. These
intervals use 3,000 resamples of 14 sources, averaging three forecaster
seeds within source. The 102 views repeat 34 recordings.

The hybrid fails the registered policy screen. The case for training
a new hybrid head is weak: its eight slots retain fewer early
opportunities than the control in both reported cohorts, and the
frozen head recovers no additional useful early choices.

## Reproduction

The intervention and evaluation were registered in `4b4edfc1`.
No models were trained this round. The previous forecast bank and
source-held selector heads are verified caches; the one-factor
replays, donor calibration, and aggregate evaluation are fresh.
With the private banks and fold checkpoints available, run:

```sh
.venv-pytorch/bin/python scripts/run_m3w_shortlist_exclusions.py freeze
.venv-pytorch/bin/python scripts/run_m3w_shortlist_exclusions.py score
.venv-pytorch/bin/python scripts/run_m3w_shortlist_exclusions.py replay
```

The replay reproduces the aggregate analysis hash. No deployment
change or independent outcome access. Neither native image distances
nor annotation steps imply meters, seconds, or physical safety.
Stage5C and SMC remain off.
