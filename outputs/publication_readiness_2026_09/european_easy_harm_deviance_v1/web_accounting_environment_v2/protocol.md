# Same-Allocation Accounting Environment Check

The first web worker ran and produced a byte-verified metadata report, but all
Slurm database observations failed. The exact errors describe unresolved socket
address families and database connection failure. The web UI's Completed label
is not accepted as evidence that the queried historical jobs exited successfully.
All 145 metadata files still match the previously collected bytes.

Hypotheses remain distinct: module-loader interference versus a compute-node
database access or resolution problem. This follow-up performs no training or
numerical inference. In one new 1-CPU, 1-GiB, five-minute allocation, it compares
system Python with process-local loader variables cleared against the same
Python module environment used by the first worker. Both branches query only
the same three owned historical job IDs, with and without the explicit cluster
flag. Three observations per branch have individual 25-second limits.

Only the diagnostic process environment changes. No shared modules, system
configuration, security settings, original jobs, trained models or data roles
are changed. Each branch writes an exclusive report with its node, interpreter,
commands, exact responses and source hash. The second branch requires the first
branch's report from the same allocation. Repeated execution is rejected.

The previous metadata report's SHA256 is
fb6465cf9782bba8d2ac29e40d9b5f55ca5b2c6a8137b198bf10f90a91fd97a3.
A successful new observation may supplement that verified metadata, but cannot
rewrite its failed queries as successes. Acceptance still requires a real exact
COMPLETED/0:0 join row and unchanged original complete-grid admission. A failed
comparison remains a transport blocker, not a negative model result.

Stage5C and SMC remain disabled. Scientific readout remains not_run.
