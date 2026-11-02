# Calibration Execution Checkpoint

## Completed

Fresh_run: complete conversion/replay of the originally reserved 102 recordings
in 12 locality groups; 26,112 queries at the registered cap of 256 queries per
recording. There are 481,537 past-supported targets and 801,360 visible-agent
rows, including 319,823 partial-history context rows. Input and bridge checks
each independently verify 306 prefix queries. This is the complete registered
cohort and budget, not every possible raw window.

Fresh_run: all nine frozen Torch pipelines across all 102 recordings, yielding
4,333,833 repeated model-target predictions. All 918 forecast and cost outputs
were freshly recomputed and exactly matched. Those repeated views are not
independent observations. Cached_verified: existing SOURCE model weights,
normalizers and source-only cuts. One current-run largest-record pilot view
was reused during inference; the separate replay recomputed every view.
No optimizer updates occurred.

|Phase|PID|Seconds|Peak RSS bytes|
|---|---:|---:|---:|
|Largest-record pilot|4339|49.672492|2365243392|
|Complete conversion|4554|204.410813|2762833920|
|Complete raw/input/bridge replay|5028|207.244544|3306864640|
|Frozen forecast/cost inference|5393|827.282159|2431598592|
|Complete fresh forecast/cost replay|6966|837.194931|2397880320|
|Real atomic recovery/direct grid pilot|6747|69.347075|656785408|
|Complete atomic allocation|8905|12106.168735|853803008|
|Targets after complete freeze|23238|162.685442|3451518976|
|Complete calibration scoring|23672|989.094563|3310239744|
|Complete metric/feasibility replay|25146|1066.039289|3259449344|

The legacy outer two-file checkpoint crash window was reproduced in a test.
Prospective storage-v2 uses a single fsynced atomic prefix. The real largest
record controlled stop after one committed multiplier, checkpoint reload and
direct full-grid run match exactly, including the original pilot. This is not
an OS-kill test. No predictor, grid, objective, solver or risk rule changed.

The earlier pre-readout set had33checks in1.24s. The final focused set has
56passing checks in3.16s; the full repository suite was not run. The
cohort SVG is rendered and visually checked; two renders give the same hash
using only public input metadata. One locality accounts for 86.35% of targets.
All 12 groups remain; window count is not statistical independence.

## Complete Allocation And Label Freeze

The complete allocation finished with exit0 on 2 November, PID8905:
12,106.168735s, peak853,803,008B. All918recording/model full grids are frozen,
with all13multipliers and4policies. The label-free choice/solver accounting
is complete; it reads no outcomes and is not a predictive or safety result.
Nonoptimal/numerical cases retain a feasible independent parent.

All102future calibration target sets were then constructed after the complete
freeze, PID23238:162.685442s, peak3,451,518,976B. Original inputs were rechecked
against the label-builder's past interface. These calibration outcomes are now
opened for their registered purpose; they are not an untouched test cohort.

## Completed Readout: Unsupported Calibration

Complete scoring and replay exited0. All47,736metric groups are exactly
reproduced, with frozen decision feasibility and per-query matched counts
rechecked. This is not MILP reoptimization. No common multiplier passes the
original full-record contract. No process from these calibration phases is
still running; do not rerun immutable completed phases as new experiments.

The full taxonomy, all39direct paired contrasts, post-readout native selected-
cost attribution and three readout figures are complete. The attribution
reconciles23,868metric views, two known-risk ratios each; it does not fit or
change a policy. Figure PNG siblings were inspected and two SVG renders have
identical hashes. All104public locality point/CI pairs and39paired contrasts
reconstruct exactly in focused tests; this is numerical reporting evidence,
not independent model confirmation or a cold-start training reproduction.

## Remaining Research

The selected easy-harm prediction error needs a SOURCE-side diagnosis before
another loss/model change. The new descriptive findings are recorded in the
working paper addendum; they have not been typeset into the frozen v8 PDF.
Final confirmation and DroneCrowd remain closed. No learned policy is promoted.

Runtime checks, input replay and numerical equality are not predictive
improvement, a safety pass, an independent efficacy result, a risk certificate,
submission readiness or deployment approval. Prior negative model-selection
results remain unchanged. Original 2% all/easy requirements and finite support
remain. Empty support is not a safe zero.

Reporting/statistic reconstruction commands (completed phases need not rerun):

```bash
.venv-pytorch/bin/python scripts/report_m3w_independent_risk_calibration.py
.venv-pytorch/bin/python scripts/plot_m3w_independent_risk_calibration.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_completed_calibration_evidence.py
```

The label operator refuses access until all 918 full-grid policy matrices
freeze. This run found no supported common multiplier. Global-floor fallback
remains separate, not a learned safe success.

## Unchanged Boundaries

Obs8/pred12/raw stride12; local image pixels; released detector/tracker silver,
not human gold. Available-label ADE is not full12 ADE. Completion bounds do not
recover missing steps inside partial trajectories. Endpoint FDE is diagnostic
where labels are missing. Past-only released-series conditioning is not proof
of online sensor causality. No metric, seconds, true3D or foundation claim.
Original role manifest and SOURCE/selection accessors remain unchanged.
Stage5C and SMC stay off. Raw data, inputs, features, models, choices and preview
PNG remain private. Only own code, configuration, scalar receipts and derived
SVG are eligible for Git. The overall research goal remains active and incomplete.
