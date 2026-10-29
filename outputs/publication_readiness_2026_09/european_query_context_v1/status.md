# Frozen-Encoder Diagnostic: Preserved, Not Promoted

v1 completed36real Torch output-head fits,72,000updates (PID58797 exit0), but
the copied risk encoder unintentionally inherited its frozen base's gradient
flags. Each arm trained130parameters. This does not complete the intended
trainable-encoder comparison. No held readout was performed and no model was
chosen from held outcomes. The checkpoints/receipts remain intact.

See [the pre-readout repair](../european_query_context_v2/encoder_repair.md).
Corrected training restarts from the original shared-utility anchor in a new
v2 directory. Prior protocol, receipts and historical code a3c8699d are retained;
the current CLI targets v2. Independent roles remain closed. No deployment,
Stage5C or SMC change.
