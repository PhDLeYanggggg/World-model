# Complementary Forecasts Recover Most, Not All, Recording Support

I kept the CV reference and original 2% recording conditions fixed. On the
same 34 recordings and three forecast seeds, I gave the offline label-aware
oracle the pre-existing complementary forecasts, one possible path per row.
This is a retrospective bank ceiling, not a trained selector or calibration.

| Alternatives to CV | Feasible / 102 | Feasible and positive | Full-roster utility |
|---|---:|---:|---|
| EqMotion | 91 | 91 | Not estimable under full support requirement |
| EqMotion + PCT | 93 | 93 | Not estimable under full support requirement |
| Both + five other fixed causal rollouts | 101 | 101 | Not estimable under full support requirement |

The final unsupported view is recording 10, seed 43, source 053. Two
observed easy rows supply positive CV reference, but no distinct path among
the seven alternatives can supply a selected easy reference within 2%.
The full candidate bank therefore **still cannot pass all views**, even
with future labels. Removing that view or treating CV equality as a safety
certificate would change the protocol; neither was done. The 101 views are
repeated measurements of 34 recordings, not 101 independent sites.

PCT alone is a weak average comparator, but has complementary choices under
an oracle. Adding it supplies two further feasible views; the physical
rollouts supply eight more. That isolates a candidate-availability problem
without demonstrating learnable routing. The next worthwhile model test is
to train a candidate that preserves the strong causal path on easy histories
while improving difficult motion, using fitting-only targets and the same
held budget. Re-tuning the ranking of the unchanged binary bank cannot cure
the eleven infeasible views. No result here licenses a smaller test roster.

```sh
.venv-pytorch/bin/python scripts/run_m3w_recording_bank_ceiling.py run
.venv-pytorch/bin/python scripts/run_m3w_recording_bank_ceiling.py replay
```

All 102 binary controls, the nested-bank outcomes and original-evaluator
checks replay exactly. Both runs exit 0 (5.75/5.74 s). New arithmetic on verified
paths, no fit or inference, no saved oracle selections. The task remains
image-coordinate native 8/12. No DroneCrowd labels, independent evidence,
deployment, Stage5C or SMC.

[Full scalar results](analysis.json) | [Registered protocol](protocol.md)
