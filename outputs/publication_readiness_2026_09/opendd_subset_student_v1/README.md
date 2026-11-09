# Learning Which Edits Can Belong to a Feasible Subset

In the [optional-edit study](../opendd_optional_edits_v1/README.md), an
outcome-aware solver found risk-feasible, beneficial subsets of frozen proposals
in 179 development recording-seed views. A causal moment gate rarely
found them. I trained three otherwise identical binary heads to see
whether the missing information can be recovered from the observed
motion and proposed path.

## Different Supervision, Same Inputs

1. **Local gain:** Does this edit improve conservative accuracy versus
   CV on its own?
2. **Optimum membership:** Is this edit in the donor recording's
   optimal optional subset, under the original risk constraints and
   positive lower gains against both CV and OLS4?
3. **Forced feasibility:** If the solver must include this edit, can it
   still find a positive, risk-feasible subset? This labels admissible
   alternatives, not just the members of one optimum.

The teacher's objective and constraints match the previous subset
proof. Every positive witness is verified with the original risk
evaluator. Unresolved solves are masked. Donor proposals come from
past hindcast nominations and a medoid of legal paths; their neural
candidate producer excludes the donor and the evaluated location.
This donor proposal rule is not the held parent policy. It avoids
using a donor proposal model exposed to the held source, at the
cost of a donor-to-held proposal shift. The held student keeps or
drops only original parent edits at fixed zero logit. The `both`
variant requires positive predictions from both recording teachers.

## Can the Labels Be Predicted?

21 folds (seven locations, three seeds) each fit three independent
heads with identical initialization and batches. The teacher sees
23,418 donor action examples across 126 donor-fold cells, repeated
across folds rather than independent. It marks 6,784 as members of one
optimum and 10,114 as forced-feasible, with no unresolved donor solves.
At evaluation, the frozen parent offers 3,690 edits. The solver's
1,303 positive optimum-member labels replay the previous proof. One
forced-feasibility label is unresolved and excluded from that probe.

| Target | Positive fraction | AUROC | AP | Precision at the fixed threshold |
|---|---:|---:|---:|---:|
| Local gain | 0.477 | 0.553 | 0.514 | 0.514 |
| Optimum membership | 0.353 | 0.507 | 0.354 | 0.363 |
| Forced feasibility | 0.523 | 0.563 | 0.584 | 0.599 |

The three rows have different targets. On the **same forced-feasibility
target**, the optimum-membership head scores 0.541 AUROC and the local
head 0.521. Thus the broader labels have some transferable signal in
this development comparison, but they do not identify a jointly valid
set. Two edits that are feasible separately need not be feasible
together. Optimum membership is especially weak in the two sites that
provide most offered edits (0.458 and 0.543 AUROC).

## Does It Improve the Decisions?

| Keep/drop rule | Edits | Edited views | All risk checks pass | Risk pass and positive vs CV/OLS4 | Signed easy >2% |
|---|---:|---:|---:|---:|---:|
| Frozen pair policy | 3,690 | 439 | 47 | 41 | 164 |
| Previous moment gate | 282 | 128 | 35 | 15 | 3 |
| Local gain student | 2,014 | 378 | 38 | 31 | 86 |
| Optimum member student | 793 | 181 | 11 | 10 | 37 |
| Forced-feasible student | 1,344 | 224 | 13 | 11 | 52 |
| Both recording heads | 776 | 175 | 11 | 10 | 37 |

None of the students has a conservative CV gain interval above zero.
The optimum student reaches +0.147 pp [-0.066, +0.496], but passes
only 11 recording-seed views. Its own released radii give identical
additive and joint outputs, so there is no additional joint lift.
All counts are from the same 606 development views, representing
202 distinct recordings. Zero edits do not count as risk passes.

**Ranking check, without a threshold.** I froze the previous gate's
282 released edits and exact per-query radii. Each student ranks the
same legal slate menu by summed action logits. The old policy passes
35 views; local, optimum and forced ranking pass 37, 34 and 33.
The local ranker finds 10 strict positive pair views against the old
policy, but the average conservative pair gain is negative for
every student. For forced feasibility it is **-7.871
[-22.451, -0.118]** in 10^-3 native units. Conservative individual
gain is -0.273 pp [-0.821, +0.017]. Intervals use 3,000 location
resamples after averaging seeds within recording (four locations in
the matched release). This rules out blaming the failure solely on a
release threshold in this experiment.

The result does not support deploying teacher imitation. Unlike
[decision-focused ranking](https://proceedings.mlr.press/v162/mandi22a.html),
the heads do not optimize the downstream regret of a complete slate.
An immediate methodological gap is the difference between individual
compatibility and joint decisions; an evidence gap is the lack of a
new, adequately labelled development source. Neither is solved by
resampling the same OpenDD recordings or adding a larger action MLP.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_subset_student.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_subset_rank_replay.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_subset_student.py score
.venv-pytorch/bin/python scripts/run_m3w_opendd_subset_rank_replay.py score
```

Native arm64 CPU, four threads, no data workers; 500 steps per fold
and atomic checkpoints every 50 steps. The private experiment
directory contains the PID heartbeat, teacher cache and score folds.
Rerun the interrupted command to resume. The run uses 8 observed
positions and 12 forecast steps in native coordinates. No new CREATE
job, sealed outcome access, deployment change, Stage5C or SMC.

[Registration](registration.json) | [Training freeze](freeze.json) |
[Keep/drop readout](readout.json) | [Matched ranking](rank_readout.json)
