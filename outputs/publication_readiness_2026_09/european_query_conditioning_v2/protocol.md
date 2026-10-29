# Query-Head Conditioning v2: Numerical-Check Repair Only

This inherits the entire scientific protocol of
`../european_query_conditioning_v1/protocol.md`, unchanged. The original v1
registration and incomplete receipts are preserved. The prior attempt stopped
at a float32 gradient-additivity assertion; same-batch double precision
reproduced the decomposition to2.89e-16 relative error. See the bound v1
interruption and numerical reproduction records.

Only the numerical verification is repaired: native component-gradient sum
norm error must be <=32 float32 eps times the sum-of-component-absolute-values
norm; all288 batches additionally verify a separate double-precision copy at
1e-10 relative norm error. Real gradients, disposable updates, optimizer history,
losses, original model weights and thresholds remain in their original precision.
No learning algorithm change or clipping choice is made.

Run all72 fitting views,36 head groups,288 matched query batches and576 disposable
AdamW steps. Compare original clip5 against no clipping on the same fixed batches.
No persistent training, held inference, independent roles, new horizon or risk
criterion, deployment, Stage5C or SMC. The primary common cohort remains frozen
self selection. Unknown outcomes remain in causal inputs and derivative counts.

Commit the repaired code, tests and registration before full rerun. Verify old
completed numerical receipts exactly after dropping only the new additivity
receipt. A one-step fitting improvement cannot establish held generalization.
Do not promote fitting diagnostics to world-model or paper success.
