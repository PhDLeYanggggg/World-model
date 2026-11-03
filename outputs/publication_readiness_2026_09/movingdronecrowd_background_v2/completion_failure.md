# Completed Pair Work, Failed Summary Assembly

The full v2 matching process (PID 69529) exited with code 1 after its heartbeat
recorded all 20,580 pairs at 2026-11-03 14:47:41 UTC. The completed pair loop took
4,136.420 seconds, with 674,234,368 bytes peak RSS. It generated 20,530 new pair
receipts and reused the fixed 50-pair pilot. All 20,580 files remain intact.

The traceback is in the subsequent summary, not matching, download or disk use:
`dict() got multiple values for keyword argument 'scientific_roles_assigned'`.
The grouping helper already returns that field as false; the summary then
expands the helper result and explicitly passes the same false field again.
Python rejects the duplicate keyword. A focused regression reproduces this
failure without recomputing image matches.

The failed original definition, log and pair receipts are preserved. Log SHA-256:
`184a07ea636b88e0c2e2c9899561498753f37f85b0f359562a7cbc83d6ffe7ef`.
No original scientific file, registration, pair, threshold, group constraint or
protected data role is edited. No successful original execution is invented.

The forward completion adapter removes only the duplicate dictionary field
during assembly, retains the final explicit false field and requires every
grouping field to equal the original helper output. It restores the original
process-local helper even on failure. Its definition is committed and registered
before recovery. The registration binds the complete pair-file digest inventory,
original registration, failed log and final original heartbeat. Recovery reads
all cached work; it does not repeat the 69-minute matching job. A separate
verification re-extracts all 360 prepared feature views, verifies the 336 old
views and all 20,580 pair receipts, then freshly recomputes the same fixed first
50 pairs. That is not a new full matching run or new source-admission decision.

This repair has no training, predictive-error readout, role assignment, threshold
selection, metric/seconds claim, Stage5C execution or SMC. The general research
goal remains active and unmet.
