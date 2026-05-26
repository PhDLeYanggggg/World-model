# Stage42-BF User Action Required

- source: `fresh_in_memory_schema_conversion`

## write_non_git_feature_store_for_stage42_bg

- priority: `high`
- notes: The in-memory conversion and baseline audit passed. Next step is to write a non-committed feature store/cache and run t100 policy source-CV.

## keep_t100_claim_blocked

- priority: `high`
- notes: This audit computes causal baselines and source-CV baseline selection, but it does not train/evaluate the protected M3W policy on these sources.
