# Scheduling Both Intervention Roles

I tested whether spending a small easy-support edit before or at
the same query as the large gain edit helps. The online rule uses
the first available easy anchor and does not force release when a
recording ends. A learned variant also requires the frozen local
support head to predict probability at least 0.5. The trajectories,
heads and original 2% evaluation rule are unchanged.

The timing control retains the same support recipient, time and
amplitude and the same per-recording role counts. When the online
policy uses gain, the control uses the earliest positive gain row
distinct from the fixed support recipient. Its knowledge of which
roles the online rule ultimately spends is retrospective. It is
a paired control, not a deployable competitor.

| Policy | Edits | Undefined | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: |
| Original independent two slots | 943 | 295 | 172 | 138 |
| Online support first | 679 | 338 | 135 | 100 |
| Same support, matched earliest gain | 679 | 346 | 139 | 110 |
| Learned support first | 30 | 596 | 4 | 4 |
| Learned support, matched earliest gain | 30 | 597 | 3 | 3 |

The primary matched contrast is **-0.01466 pp [-0.07269, +0.04923]**.
The support-first rule delays 250 of 331 gain actions. Requiring the
learned local-support probability leaves just 15 paired actions
(30 total edits). The 679-edit policy also loses to the old
943-edit independent policy in the full comparison, but the counts
differ. Intervals average three seeds then bootstrap seven sites
3,000 times; 606 views are 202 previously exposed recordings scored
with three seeds, not independent confirmation.

## Exact Capacity

These are outcome-aware limits, not models. The finite search uses
the unchanged risk scorer and conservative gains against CV and OLS4.
Every positive witness in the timing and shape searches was rescored.

| Permitted changes | Joint feasible out of 606 |
| --- | ---: |
| Old first action fixed; second action anywhere (previous audit) | 255 |
| Both roles anywhere; each recording retains original role counts | 473 |
| Same counts; support no later than gain | 470 |
| Support-first realized counts; both roles anywhere | 310 |
| Support-first realized counts; support no later than gain | 308 |
| Any counts up to one edit per role, neural directions | 485 |
| Same cap and per-row radius; also allow OLS4 direction | 511 |
| Same cap and per-row radius; also allow observed-prefix direction | 515 |
| Same cap and per-row radius; both extra directions | 518 |

With the neural bank, 541 views have a risk-feasible action but
only 485 can also beat both baselines. The 121 failures separate
into 30 with no full easy reference, 35 with reference but no
risk-feasible action, and 56 that have risk-feasible actions but
cannot beat both baselines under risk. Of those 56, **52 can beat CV
under risk but cannot beat OLS4**. The analytic candidates add
33 verified witnesses without introducing new editable rows or
amplifying analytic displacements.

The larger timing capacity mainly comes from releasing **both**
roles from the old decisions, not from the support-first ordering.
The next experiment should compare a wider causal forecast bank
with neural-only at matched budgets, with supervision relative to
both CV and OLS4. No new ordering heuristic can make an absent
reference defined. Nothing here changes the original risk rule.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_support_schedule.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_schedule_capacity.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_schedule_shapes.py
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_calibration_roles.py verify
```

`fresh_run`: causal policy readout and three exact feasibility
comparisons. `cached_verified`: frozen forecasts, model weights and
prior results. No new fitting, threshold tuning, CREATE job,
deployment change, Stage5C or SMC. `not_run`: independent confirmation;
the 15/30 DroneCrowd allocation remains frozen.
