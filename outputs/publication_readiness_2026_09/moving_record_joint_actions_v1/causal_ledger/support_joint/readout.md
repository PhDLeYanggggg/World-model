# Equal-Budget Support Retrieval And Ordered Joint Repair

The previous label-aware capacity test found 33 of 35 views with
undefined easy support repairable by one insertion followed by
optional deletions. The old eight-slot list retained only 18.
Here both retrieval rules mask already-switched rows before
filling the same eight slots. The control preserves family
diversity; the alternative alternates predicted easy credit,
easy reference, joint easy/safe score, and gain. The original
forecaster bank and causal prediction heads remain frozen.

| Candidate set | Checked ordered feasible / 35 | No eligible anchor | Infeasible |
|---|---:|---:|---:|
| Eligible family-diverse list | 18 | 12 | 5 |
| Support-specific list | 20 | 14 | 1 |
| Whole eligible bank, prior capacity | 33 | 0 | 2 |

The support list gains two feasible views in aggregate but leaves
more views with no eligible anchor; it is not a uniform improvement.
These are checked label-aware capacities, not policy passes.

## Operating Policies

Each candidate policy uses only current and past data to decide
whether to add an action. Joint versions may remove one old action
in the same query or after insertion. Both predicted risk budgets
and paired gain are checked on the proposed change; the original
query cap stays fixed. No new prediction or scoring head is fit.

| Policy | Cells with donor-selected cuts / 42 | Insertions | Deletions | Complete passes / 102 | Signed easy >2% |
|---|---:|---:|---:|---:|---:|
| Direct | Not applicable | 0 | 0 | 2 | 8 |
| Family retrieval and joint edits | 0 | 0 | 0 | 2 | 8 |
| Support retrieval and insertion only | 0 | 0 | 0 | 2 | 8 |
| Support retrieval and joint edits | 0 | 0 | 0 | 2 | 8 |

No arm changes a held decision. Paired differences are mechanically
zero; no intervention efficacy contrast is estimable. All arms have
35 undefined easy upper bounds. This is a failed operating experiment,
not an identity-based risk certificate.

## Calibration Failure Is Upstream Of Recording Risk

Across the fixed donor grid, no cell for any arm recovers a
single complete pass. Both joint policies propose edits to
19 of 190 repeated missing-support donor occurrences, but
none of those occurrences receives an observed easy-reference
action under any tested cut. The insertion-only variant proposes
none on that stratum. The counts are unions across the grid
per donor occurrence, not independent observations or a
new selected threshold. No held outcome contributes to this
calibration diagnostic.

The frozen scores can propose some edits, but their selected
actions do not supply the support they are meant to add.
Additional threshold tuning on these scores cannot fix that
observed failure within the registered grid. A better next
target is observed easy-reference precision on reachable
insertion candidates, measured separately from paired gain
and the final recording risk.

## Reproduction

Registration `e78fb64f` precedes donor calibration and decision
freeze. Fit, calibration and held roles match the prior 42 cells:
seven, six and one source, respectively. With private forecast and
component checkpoints available:

```sh
.venv-pytorch/bin/python scripts/run_m3w_support_joint.py freeze
.venv-pytorch/bin/python scripts/run_m3w_support_joint.py score
.venv-pytorch/bin/python scripts/run_m3w_support_joint.py replay
.venv-pytorch/bin/python scripts/audit_m3w_support_joint_calibration.py
```

Freeze resumes at completed source checkpoints. Replay recomputes
the observed scores and checked list capacities. The donor audit
does not use held outcomes for threshold selection. The task is
exposed Moving A development, eight observed and twelve future
annotation steps, native image coordinates, and 34 recordings
repeated over three forecaster seeds. No independent outcomes,
deployment, metric or seconds claim, physical safety, Stage5C or
SMC.
