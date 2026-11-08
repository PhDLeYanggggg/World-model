# Learning Fixed-Budget Shape Choice on TUMDOT

This test follows the [matched frozen transport study](../tumdot_matched_finite_transfer_v1/README.md).
The same six forecast directions, recipients and per-row peak radii
remain fixed. I train a selector on four donor recordings for each
held recording, excluding both its intersection and collection day.
No exact released track identifier is shared with the four donors.
The nine recordings are previously exposed, from one campaign at
three intersections; this is development, not independent transfer.
The [TUM release documentation](https://www.mos.ed.tum.de/en/vt/forschung/datensaetze/tumdot-ing/)
describes aerial trajectory extraction and overlapping flights. The
past-only checks apply to our use of the released positions, not
to a verified forward-only upstream detector/tracker.

Inputs use eight observed positions, relative forecast geometry,
and optional short backtests against already observed portions of
the history. Targets use 12 future annotation samples only for
learning or scoring. The head predicts per-step conservative gain
against CV (including exact disagreement debit at missing steps)
and known all/easy positive harm. Donor samples, initialization and
capacity match with and without past backtests. The gain-only and
risk-penalized decoders share a head. Quarter radius is a predeclared
scoring sensitivity of the half-radius model, not a second fit.

| Half-radius rule | Lower utility vs CV (pp) | Risk passes / 27 | Risk and positive vs CV and OLS4 |
| --- | ---: | ---: | ---: |
| Fixed past-motion hindcast | -0.03654 | 24 | 3 |
| Past backtest only | -0.04003 | 24 | 3 |
| Learned, with backtest features | -0.04273 | 22 | 6 |
| Learned, without backtest features | -0.02634 | 22 | 5 |
| Risk-penalized | -0.04372 | 22 | 7 |
| Outcome-aware menu witness | +0.42447 | 27 | 27 |

The learned backtest arm is +0.00279 pp versus hindcast on observed
steps but -0.13422 pp when missing-step disagreements are charged.
The directly paired backtest ablation favors the no-backtest head
at each of the three intersections (-0.03002 pp for adding it).
No learned arm has positive mean lower utility versus CV. At
quarter radius all arms pass risk on 27/27, but all causal arms
still have negative mean CV utility. The oracle is not a model.

Descriptive bootstrap intervals and per-intersection effects are in
[analysis.json](analysis.json), along with native-coordinate ADE/FDE,
risk and easy metrics. Three intersections in one campaign are not
strong population inference. The subsequent [paired-target trial](../tumdot_incumbent_selector_v1/README.md)
tests whether scoring a change directly against the fixed hindcast
reduces its disagreement cost without moving the edit budget.

```sh
.venv-pytorch/bin/python scripts/run_m3w_tumdot_matched_selector.py
.venv-pytorch/bin/python scripts/run_m3w_tumdot_matched_selector.py --replay
```

`fresh_run`: 54 fits and 21,600 updates, 16.999 s, PID 90430.
`cached_verified`: frozen source forecasts, exact controls, zero-update
replay. Per-recording feature checkpoints and per-model checkpoints
every 100 updates, arm64 four threads, zero loader workers. No CREATE
job, deployment, Stage5C, SMC or independent confirmation.
