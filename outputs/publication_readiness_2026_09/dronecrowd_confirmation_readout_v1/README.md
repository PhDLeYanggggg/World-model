# One-Shot Confirmation Readout: Registered, Not Run

The [protocol](protocol.md) and [registration](registration.json) fix
the outcome analysis before calibration is scored. The 72 confirmation
recordings remain unopened by this runner. The existing local
40-recording calibration job was not modified or restarted.

The readout separates two questions. First, can the selected policy
improve conservative forecast utility versus CV and OLS4 while
meeting the original recording risk rule? Second, are any gains
specific to the joint choice, rather than which queries were selected
or how many agents were switched? Independent, shared-family and
pair-additive controls share the selected policy's active-query counts.
An empty same-count comparison stays undefined; those controls do not
replace the original policy's risk check.

Admission verifies the replayed calibration result and frozen forecast
receipts, recomputes the registered winner, and refuses to proceed
without a qualifying choice. Before any outcome access, confirmation
predictions must be frozen across the entire approved roster. The
one-shot scorer writes resumable per-record aggregate statistics;
replay uses those statistics without rereading labels. Available-label
ADE and available-endpoint FDE report their support explicitly.

Current verification is limited to registered code and synthetic
checks, not confirmation inference. Run the metadata-only check with:

```sh
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_confirmation.py register
```

If calibration qualifies, the next step is to freeze its admission
receipt and pick an execution location for the full confirmation roster.
At the measured serial calibration CPU rate, the 2.76 million
confirmation targets would take roughly 13 hours. A separate
[development benchmark](../native_cpu_shard_benchmark_v1/README.md)
supports an optional process-isolated local backend. It changes only
the scheduling of inference, and remains blocked on calibration.
No deployment, Stage5C or SMC.
