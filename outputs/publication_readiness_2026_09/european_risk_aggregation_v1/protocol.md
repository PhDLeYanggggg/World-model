# Frozen TRAIN Risk Aggregation Diagnostic

The preceding synthetic counterexample proves that query-mean feasibility does
not imply agent-mass feasibility. This diagnostic measures that discrepancy on
the existing fixed 128-query TRAIN monitors, not on new development predictions.
All 144 completed scaling/unit checkpoints are required. Inputs, eligibility,
sampling seeds and saved soft-excess monitors must replay. No optimizer step,
model choice, threshold search or independent role access is allowed.

For soft and original hard actions, report both the query-mean excess and the
pooled selected-agent excess H - .02 R (all and easy separately). The zero sets
are compared before division; zero selected reference has undefined risk and is
not evidence of safe support. Only known TRAIN outcomes enter this diagnostic;
it cannot assess unknown-outcome completion bounds. Repeated sampled queries are
not independent evidence, and these monitors are not the entire TRAIN population.

Count both disagreement directions, positive-excess views, and undefined support.
This is a posthoc mechanism check of frozen models, not an intervention isolating
causation. It cannot prove which factor caused development failure, prescribe a
test threshold, or justify changing the registered scientific risk definition.
The active scaling readout remains untouched. Stage5C and SMC stay disabled;
dataset-local/raw-frame claims and exploratory provenance remain unchanged.
