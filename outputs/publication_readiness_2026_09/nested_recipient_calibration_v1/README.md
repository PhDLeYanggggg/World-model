# Nested source-disjoint recipient calibration

**Status: fresh development experiment, no eligible policy.** This is not
the DroneCrowd 15/30 experiment. Its official calibration and independent
confirmation outcomes were not accessed in this run.

## Design

The 69,746 fitting rows span 24 released MovingDroneCrowd++ recordings
and nine original fitting source groups. Each outer fold separates six
fitting groups, two inner calibration groups, and one outer scoring group.
Both rankers share a fresh fit-only ridge path nominator over the same ten
paths. The control ranks by nominated gain. The neural arm ranks the
probabilities of positive gain, harm or missing support, and safe-easy
outcomes. Both use the same past-only query quota.

I fitted 27 Torch heads (seeds 17, 29 and 43, 500 updates each) on native
arm64 with four CPU threads and zero loader workers. The inner threshold
must pass the unchanged risk rule on every inner recording and yield
positive conservative utility against both CV and OLS4. If none qualifies,
the policy falls back to CV; undefined easy risk does not pass.

## Results

| Check | Result |
|---|---|
| Inner grid selected | 0/27 fold-seeds for each ranker |
| Calibrated outer policy | Zero switches; easy risk undefined in all 72 recording-seed views |
| Fixed quota | 95,769 switches per ranker; 0/72 original risk passes each |
| Fixed support minus fixed gain | -0.701 pp conservative gain, 95% CI [-1.709, +0.381] |
| Fixed support minus OLS4 | -7.813 pp conservative gain, 95% CI [-14.768, -0.919] |
| Exact inner score frontier | 191,505 support and 191,292 gain levels; no full risk pass |

The intervals use 3,000 paired source bootstrap draws after averaging the
three training seeds. The zero-switch matched efficacy comparison is not
estimable. The full frontier is a diagnostic on the inner recordings and
did not change any frozen threshold or outer choice.

Four of nine inner splits contain a recording without any easy examples
in the entire roster. In the other five, the same nominees admit a
**risk-only hindsight witness** in every inner recording: one eligible
easy action meeting the individual 2% reference bound, optionally
accompanied by nonharmful beneficial actions. A weighted average of
nonnegative harm cannot satisfy the bound if all eligible easy actions
exceed it. The witness uses labels and does not establish a joint gain
over both references. Missing easy support and inability to rank available
actions are separate failure modes. A finer score grid is not a remedy
for either one.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_nested_recipient_calibration.py
.venv-pytorch/bin/python scripts/run_m3w_nested_recipient_calibration.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_nested_recipient_frontier.py
.venv-pytorch/bin/python scripts/audit_m3w_nested_calibration_support.py
```

The fresh process saved every fit and outer score before a report-only
function reference failed. The [patch receipt](reporting_patch.json)
binds the correction; replay rebuilt the summary without refitting or
changing actions. Local checkpoints and row predictions are not distributed.

[Registration](registration.json) | [Freeze](freeze.json) |
[Analysis](analysis.json) | [Frontier](frontier.json) |
[Support audit](support_audit.json).
