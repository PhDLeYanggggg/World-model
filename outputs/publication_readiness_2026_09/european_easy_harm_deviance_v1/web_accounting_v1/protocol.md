# Read-Only Accounting Through Open OnDemand

This is an execution-only metadata collection, not a new model experiment.
Use the official authenticated CREATE Job Composer in a new M3W-owned workflow.
The existing C7 workflow and other projects are outside the mutation scope.

The worker requests one CPU, 1 GiB and exactly five minutes, with no requeue.
It requires a CREATE Slurm compute allocation before accessing the owned M3W
experiment. Scheduler queries are limited to recovery37835856, join37835859 and
portability37837636. Each query has a 35-second timeout; failed observations remain
unknown and are not retried as submissions. No training, numerical inference,
checkpoint reads, independent labels, cancellation or network-setting changes
are allowed.

The original registration and recovered freeze hashes are fixed. The worker
reads the existing 144 fit JSON receipts and verifies their path and byte hashes.
It writes a new exclusive web_accounting_v1 directory in the owned experiment;
an existing directory blocks repeated execution. Original outputs are read-only.

Before submitting, commit and upload the exact worker and Slurm script into the
new workflow. Confirm its script, account and empty array field. Record the
single submission intent and resulting job ID. An ambiguous response must be
reconciled from the same workflow, never followed by a second Submit click.

After execution, retrieve the report through the official file browser. Compare
the report hash to the completed worker's stdout and its worker hash to the
committed source. Accept only the exact successful join accounting row with
exit0:0 and byte-identical metadata. Run the unchanged local complete-grid
admission. The canonical scheduler-backed collection is generated only from
those actual observations, never from a log-completion inference.

The separate six-arm scientific readout remains unrun until its original
admission succeeds. This transport does not change training, prediction,
normalization, uncertainty estimation, thresholds, data roles or deployment.
Stage5C and SMC remain disabled. No metric, seconds or physical-safety claim is
introduced.
