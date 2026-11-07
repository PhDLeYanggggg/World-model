# Support and Value for a Single Causal Insertion

**Result: better first-query choices, no whole-recording improvement.**
This is a fresh native arm64 PyTorch experiment on exposed development
recordings. The three forecast seeds repeat 34 recordings in 14
source groups (102 views). I fit 42 small heads with 220 updates
each, reusing the frozen forecast bank and source roles. The three
scoring arms share the network, candidate actions and fitting draws.

| Ranking arm | First query: observed easy reference | First query: reference and nonnegative gain | Calibration folds with a cutoff | Frozen insertions | Whole-recording passes |
|---|---:|---:|---:|---:|---:|
| Original direct policy (no edit) | not probed | not probed | not applicable | 0 | 2/102 |
| Support probability | 21/102 | 4/102 | 8/42 | 7 | 2/102 |
| Direct value head | 12/102 | 3/102 | 6/42 | 7 | 2/102 |
| Factorized support/value | 21/102 | 9/102 | 6/42 | 4 | 2/102 |

The prespecified component contrast is **+4.84 pp** for the
factorized score versus direct value (paired 3,000-draw source
bootstrap 95% interval **[+0.79, +10.32] pp**). The source-weighted
contrast is not the difference of the pooled view fractions.
This unthresholded probe does not imply a recording-level benefit.
Both arms leave 35 views without easy reference and 8 signed-easy
violations. The factorized policy's paired completion-lower change
is -0.000061 pp (95% interval [-0.000168, 0.000000]).

## Where the Improvement Stops

A label-aware postcheck confirms the earlier solver result: **22/35**
missing-reference views have a feasible single insertion. Of these,
**16** have one at the first eligible query, and **6** require a later
query. Both value scores pick a whole-record-feasible first action in
only **1/35**. The factorized arm lacks a donor-selected cutoff in
20 of the 22 feasible views; its remaining 2 do not trigger. Across
all 35 missing-reference views it makes one insertion, which supplies
no observed easy reference. Its one observed support addition overall
is in a view that already had reference support.

Predicting observed reference and gain separately helps the local
choice, but that target does not distinguish actions that recover a
whole recording. The next useful comparison is a causal prefix-aware
or joint insertion/deletion model trained against the recording
objective, with a matched one-insertion control. This diagnostic
does not select a new threshold or export oracle choices.

## Reproduction

The protocol and implementation were registered in `c2f0af51`
before fitting. With verified private inputs, run:

```sh
.venv-pytorch/bin/python scripts/run_m3w_reference_hurdle.py fit
.venv-pytorch/bin/python scripts/run_m3w_reference_hurdle.py score
.venv-pytorch/bin/python scripts/run_m3w_reference_hurdle.py replay
.venv-pytorch/bin/python scripts/audit_m3w_reference_hurdle.py
```

Fitting took about 82 seconds locally with four Torch threads and no
loader workers. Checkpoints and the fold freeze are retained locally;
only scalar aggregates and execution receipts are published. Replay
reproduces the scalar score exactly. The original 2% and support
requirements are unchanged. No independent outcomes were opened;
there is no deployment, physical safety, metric, seconds-level,
Stage5C, or SMC claim.
