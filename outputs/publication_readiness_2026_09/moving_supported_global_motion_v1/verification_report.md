# Verification And Execution Record

Fresh_run:fits,refits,choices and scores. Cached_verified:parent forecasts and
feature assets. Not_run:new forecaster training or independent confirmation.

| Phase | PID | Seconds | Peak RSS bytes | Observed result |
|---|---:|---:|---:|---|
| Prepare | 90503 | 74.695561833 | 3147841536 | exit0,five exact ID joins |
| Recovery pilot | 90653 | 43.787749042 | 2046083072 | exit0,128+128vs256exact |
| Train | 90737 | 1143.657699916 | 7477329920 | exit0,96models393216updates |
| Full refits | 93036 | 1157.498922041 | 7465222144 | exit0,all96exact |
| First infer | 94774 | not recorded | not recorded | exit1,missing checkpoint directory |
| Recovered infer | 95208 | 472.858146750 | 3086680064 | exit0,all36choices frozen |
| Evaluate | 96183 | 7.484033333 | 2369470464 | exit0,full original readout |

The [runtime repair](inference_runtime_repair.md) only initializes a directory
through a committed safe entry point. Original registered scientific files and
models remain unchanged. All36choices were committed before scoring. Parent
fallback occurs in37joint-free postchecks,31matched postchecks and6matched
solver timeouts. These are not rerun until favorable or called all-optimal.

38focused checks passed3.59s before the runtime repair;its new directory test
passes. The first posthoc diagnostic failed on JSON conversion of NumPy bindings;
recursive typed equality fixes that analysis-only bug with a regression test.
The model and primary scores do not change. Full historical suite was not rerun.

18/24augmented-local parent weight sets match exactly;six have max absolute
difference5.8562e-6. All24normalizers,samplers and masked-zero branches match.
Do not confuse that comparison with96/96exact new same-implementation refits.

Actual fitting-loss,matrix and worst-easy figures are rendered and visually
checked. IQR describes24fits per cell,not validation or independent forecaster
confidence. Images/paths/checkpoints stay private.

Replay is separate:recompute predictions,independent decisions,frozen-choice
feasibility/matched counts and metrics,without rerunning timed integer search.
Its actual status is recorded separately,never inferred from the script existing.

Actual replay PID97281 exits0 in25.494838333s,peak RSS3,693,166,592bytes.
All36prediction contexts,independent choices,frozen-choice checks,matched counts
and complete metrics are exact;every public file is unchanged during replay.
Timed integer optimization is not repeated. Final combined focused tests:
40passed in3.05s. These checks do not turn failed risk/utility into a pass.

Compiled main PDF is unchanged,sha256:
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.
The new paper addendum is not a compiled or submission-ready main manuscript.
