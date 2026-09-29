# Local Input Reconstruction After Transport Failure

The memory-only SSH reader ended before its initial handshake, before any packet
was read or any diagnostic outcome was produced (JSON decode of empty transport
output). A separate bounded CREATE observation also timed out. This is an access
observation failure, not proof that scheduler job 37602475 failed. The owned SSH
reader had already exited before a cleanup signal was attempted; no remote job
or research process was killed or restarted.

Use the already verified local input cache and frozen checkpoints to reconstruct
each of the original 288 compressed packets in memory. Preserve the exact field
order, metadata, preprocessing, actions, target construction and training-only
tail definition of the original exporter. Require byte-for-byte SHA-256 equality
with every previously committed packet-manifest entry before computing that
packet's diagnostic; any mismatch stops the readout. No row cache or checkpoint
is written. This does not reconstruct or re-audit raw video or detector labels.

Run the same registered mathematical diagnostic and full numerical replay. Write
only small aggregate summaries and receipts. Four compute threads, zero workers,
native arm64 runtime. Distinguish this fresh local computation from the pending
CREATE environment replication and independent scientific confirmation.

There is no data-role, model, threshold, denominator, budget, scientific estimand
or analysis-scope change. The existing 2% selected-positive-harm budget and all
288 views remain. Independent selection/calibration/confirmation stay closed;
Stage5C and SMC remain off. No metric/seconds, true3D, foundation or gold-label claim.
