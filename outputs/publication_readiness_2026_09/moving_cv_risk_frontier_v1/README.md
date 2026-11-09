# Can risk-budgeted proposals recover feasible decisions?

I kept the v141 paths and fitted predictors fixed. Instead of taking
prefixes of a few action rankings, I used six predicted risk/support
budgets to assemble alternatives directly from the per-agent bank.
A query LP maximizes the smaller predicted incremental gain against
CV and OLS4; bounded rounding keeps both a low-violation and a
high-gain option when those differ. Only causal predictions enter
generation. These budgets do not replace the original recording
risk and support checks.

This separates prediction from the finite decision menu, in the
spirit of [predict-then-optimize](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3922).
Unlike an exact [decomposition algorithm](https://pubsonline.informs.org/doi/abs/10.1287/opre.8.1.101),
the fixed grid does not use the recording problem's dual prices to
choose its next query option. There is no convergence claim.

## Capacity, with outcomes available

The same 24 exposed recordings, nine source groups, three seeds,
and 72 recording/seed views were used. The new solver increases
the number of query options from **79,047 to 99,542**, adding at
least one option to 3,506 of 3,528 query/seed views.

| Menu | Verified | Nonpositive bound | Risk infeasible | No easy support | Unresolved |
|---|---:|---:|---:|---:|---:|
| Original v141 options | 8 | 52 | 6 | 6 | 0 |
| Original + risk-budgeted options | 11 | 49 | 6 | 6 | 0 |
| Same paths, unrestricted per-agent assembly | 27 | 15 | 0 | 6 | 24 |

The initial rounded relaxation recovered nine expanded witnesses.
Integer feasibility recovered and rescored both remaining open
expanded views. One of those also establishes a full-bank witness
in a view that the time-limited row solver had not resolved.
Every witness satisfies the original 2% recording rule and has
positive paired completion-lower gain against both references.
The 27 are a verified lower count, not the maximum: the full-bank
search leaves 24 views unresolved.

The expansion gives +1.587 percentage points in source-balanced
verified rate [0.000, 4.762]; the full-row comparison gives +11.772
[0.000, 23.552] over the expanded menu (3,000 paired source
resamples). The three new expanded witnesses do not establish a
stable source-level improvement. They do show that query options
are not equivalent to the complete per-agent action space.

## Can the frozen scorer exploit those new options?

No new model was fitted. The same v141 joint scorer selects a new
option in 456 query/seed views. Keeping its old query counts is
a different control from letting the new menu change counts.

| Causal policy | Edits | Original risk passes / 72 | Worst known signed easy change |
|---|---:|---:|---:|
| Original | 49,734 | 3 | +126.59% |
| Expanded at original counts | 49,734 | 3 | +126.59% |
| Expanded, free counts | 65,243 | 6 | +130.95% |
| Independent at expanded counts | 65,243 | 4 | +116.52% |

| Causal contrast | Observed-only pp | Completion-lower pp (95% CI) |
|---|---:|---:|
| Expanded free vs original | +0.054 | -0.119 [-0.202, -0.044] |
| Expanded vs original, same query counts | -0.006 | -0.027 [-0.079, +0.0003] |
| Expanded vs independent, same query counts | -0.013 | -0.379 [-0.718, -0.157] |
| Expanded vs CV | +1.397 | +0.144 [-0.232, +0.601] |
| Expanded vs OLS4 | +0.796 | -53.512 [-95.188, -21.895] |

The conservative pair-distance contrast against matched independent
selection is -0.633 local pixels [-1.532, -0.115]. Calibration
qualifies 0/9. More interventions and more passing views do not
offset the worst easy change or the loss in conservative accuracy.

**Next:** train on verified full-record witnesses and comparable
infeasible alternatives from fitting sources only. Use constraints
at the recording level to guide which combinations to propose,
rather than adding more fixed query budgets. First measure recovery
of the 16-view gap between the verified expanded and full-row counts
on exposed development; do not treat those witnesses as held model
predictions or tune on the sealed confirmation.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/audit_m3w_cv_risk_frontier.py
.venv-pytorch/bin/python scripts/audit_m3w_cv_risk_frontier_integer.py
.venv-pytorch/bin/python scripts/eval_m3w_cv_risk_frontier_transfer.py
```

Menu generation, bounded capacity and causal evaluation are
`fresh_run`. Shape and query checkpoints are `cached_verified`.
Commands use local arm64 Python and four Torch threads. The two
longer jobs keep private per-cell receipts and heartbeat files;
rerunning continues from verified menu/action artifacts. A from-scratch
bounded solver rerun can change the unresolved count if timing
differs. No new CREATE run, external calibration, deployment,
Stage5C, SMC, or submission. DroneCrowd confirmation remains
`not_run` under the approved 15/30 split. Eight observed / twelve
forecast steps are in dataset-local pixels, not metric or seconds.

[Protocol](protocol.md) | [Initial capacity](readout.json) |
[Integer continuation](integer_readout.json) |
[Causal readout](causal_readout.json) |
[Working supplement](../evidence_manuscript_v142/README.md).
