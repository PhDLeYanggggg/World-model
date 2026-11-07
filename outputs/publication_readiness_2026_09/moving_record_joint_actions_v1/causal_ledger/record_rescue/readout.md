# Whole-Recording Supervision for Causal Forecast Insertion

**Result: one reference recovered, no additional recording pass.**
I trained 84 native arm64 PyTorch models across 14 source groups and
three frozen forecaster seeds, with 220 updates per model. The 102
development views repeat 34 recordings; they are not independent.
The same candidate roster and sampling draws distinguish the effect
of the supervision target from the effect of past prefix features.

| Arm | First query: whole-record non-loss rescue | Folds with qualifying cutoff | Insertions | Undefined easy reference | Recording passes |
|---|---:|---:|---:|---:|---:|
| Original direct policy | not probed | not applicable | 0 | 35 | 2/102 |
| Local supported gain target | 1/102 | 9/42 | 13 | 35 | 2/102 |
| Whole-record rescue target | 2/102 | 4/42 | 9 | 34 | 2/102 |
| Whole-record target with prefix | 1/102 | 3/42 | 6 | 35 | 1/102 |

The prespecified first-query contrast is +0.48 pp for whole-record
versus local supervision (paired source-bootstrap 95% interval
[0.00, +1.43] pp); prefix versus no prefix is -0.48 pp
([-1.43, 0.00]). Neither excludes zero. The labels are sparse:
41,549 of 4,185,718 repeated fitting-action occurrences are positive
for whole-record rescue, versus 206,129 for local supported gain.
The whole-record arm changes paired completion-lower utility by
+0.00056 pp ([-0.00058, +0.00223]). Prefix loses one original
recording pass and decreases paired lower utility by 0.02919 pp
([-0.08663, 0.00000]). The eight signed-easy violations remain
in all arms; no arm delivers a new non-loss pass.

## What Went Wrong

The whole-record arm's one recovered reference reduces the all-risk
upper ratio from 20.61% to 12.82%, still above 2%. In a label-aware
postcheck, fixing that same insertion and allowing ordered deletions
admits a verified one-deletion witness: it satisfies the original
recording checks and has nonnegative paired lower utility. It does
not follow that the deletion is identifiable from past observations.
The prefix arm's lost pass breaches both all-risk and easy-risk
upper bounds; it is not an effect of dropping empty denominators.
No choice or threshold was modified during this analysis.

This makes a joint insertion/deletion comparison more informative
than another single-insertion threshold sweep. Ranking feasible
decisions and learning constraint parameters are established in
decision-focused learning ([Mandi et al., ICML 2022](https://proceedings.mlr.press/v162/mandi22a.html);
[Mandi et al., NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/ef158a0d4f5364741a571b8d1d44fb1b-Abstract-Conference.html)).
This experiment only compares supervised scores for a fixed action
set under partial-label recording risk. It does not implement those
optimizers or establish a new general learning principle.

## Reproduction

The experiment was registered in `6a173760` before fitting.
With the verified private parents:

```sh
.venv-pytorch/bin/python scripts/run_m3w_record_rescue.py fit
.venv-pytorch/bin/python scripts/run_m3w_record_rescue.py score
.venv-pytorch/bin/python scripts/run_m3w_record_rescue.py replay
.venv-pytorch/bin/python scripts/audit_m3w_record_rescue.py
```

Local fitting took about 108 seconds with four Torch threads and
zero loader workers. Replay reproduces the original scalar score.
Checkpoints and the decision freeze stay local. The original
2% and support rules remain; no independent outcomes were opened,
no deployment changed, and no metric, seconds-level, Stage5C or
SMC claim follows.
