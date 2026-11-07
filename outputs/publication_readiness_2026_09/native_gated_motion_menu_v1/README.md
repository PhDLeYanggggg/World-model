# Learned OLS4-Relative Motion Options Do Not Expand Capacity

The [source-separated trained heads](../native_gated_motion_residual_v1/README.md)
hurt the fixed-recipient policy but improve all-row available-label
error against OLS4. This follow-up asks whether either remains useful
as an optional path. All decisions in this audit are hindsight,
not available to a deployed selector.

| Bank | Verified both CV and OLS4 | Negative maximin upper bound | Risk/count infeasible | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Native | 51 | 45 | 6 | 0 |
| Native plus ungated head | 51 | 45 | 6 | 0 |
| Native plus gated head | 51 | 45 | 6 | 0 |
| Native plus both heads | 51 | 45 | 6 | 0 |

At the same 9,675 genuine replacements, none adds a verified view,
recording or source group. All three paired source-bootstrap changes
in verified rate are 0 pp [0, 0] on this exposed roster; this is not
population equivalence. The 51 existing witnesses are re-scored
and retained, not re-optimized. The result does not exclude
better accuracy on already positive views or at a different budget.

The [protocol](protocol.md) and [registration](registration.json)
were fixed before this follow-up optimization. The CPU solve took
231 seconds and replay exactly reproduced the scalar readout by
re-scoring the witnesses. Replay does not rerun solver proofs.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_gated_motion_menu.py score
.venv-pytorch/bin/python scripts/audit_m3w_gated_motion_menu.py replay
```

[Analysis](analysis.json) | [Score receipt](score_execution.json) |
[Replay receipt](replay_execution.json). No new training, independent
outcomes, deployment changes, Stage5C execution or SMC. The task
uses native annotation steps and image pixels.
