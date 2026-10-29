# First Diagnostic Execution: Group Accounting Bug

The first execution finished at 09:06:38Z on 29 October 2026, but its learned
policy per-recording table is invalid. Numeric recording IDs were serialized as
string dictionary keys and then compared with numeric arrays, producing empty
policy slices. The overall view scores and per-recording oracle loop did not use
that faulty comparison, but no v1 aggregate is promoted as the final diagnostic.

All v1 receipts remain unchanged. A numeric-ID regression test reproduces the
failure (one failed, one passed). Version v2 normalizes IDs consistently, checks
record-count sums, and compares every per-recording frozen policy score against
the previous verified readout. It recomputes the same full roster in a new directory.
No split, objective, budget or model is changed, and no held outcomes are newly
scored. This is an engineering repair, not a scientific result or model improvement.
