# Stage42-BE User Action Required

- source: `fresh_local_conversion_readiness`

## run_stage42_bf_actual_schema_conversion

- priority: `high`
- notes: Write the actual converted external schema rows outside Git, then run no-leakage and train-only source-CV. This Stage42-BE report is readiness only.

## keep_t100_claim_blocked_until_source_cv

- priority: `high`
- notes: The novel local files can support future t100 repair, but current t100 positive claim remains blocked until converted rows pass source-CV.

## prioritize_ucy_source_cv

- priority: `medium`
- notes: The three novel UCY-like t100 sources are enough for a leave-one-source-out readiness plan after conversion.
