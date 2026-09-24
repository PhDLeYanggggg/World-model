# CREATE Observation, 2026-09-24

Result source: fresh_run, read-only queue observation. This is an infrastructure
check, not an M3W experiment or a simulation result.

The existing authorized connection and host-key restrictions were preserved.
The corrected query completed successfully and showed four running simulation
array tasks on the biomedical A100 partition and one dependency-pending CPU
task. No task was submitted, cancelled, restarted or modified. No running
simulation resources were appropriated for M3W.

The first read-only attempt failed because our SSH command had unquoted format
separators. Its failure receipt is retained; quoting the remote command repaired
the query. This was not evidence of an authentication failure or a failed HPC job.

Private evidence hashes:

- Failed command receipt: `fc89c6fa2a7d45d7e93a5e1ab7f4869f42376ac8362e1d06cbfe34462c31479e`.
- Successful queue receipt: `0370b4813e1edfaa51ae38c4ed849c5ac606cea82646ead128bbe5d5cc4aa63b`.

No M3W job was identified in this queue snapshot. Remote M3W directories and
historical outputs remain unverified; a queue snapshot cannot establish their
absence. The current bounded download and per-file parser are suitable locally,
so there is no need to disturb other work or launch duplicate HPC jobs.
