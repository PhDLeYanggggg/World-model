# Stage43-F Source-Level Split Gate

- verdict: `stage43_f_source_level_split_ready`
- gate: `11 / 11`
- next action: train/evaluate Stage43 latent model on this source-level split; old Stage43-C checkpoint remains UCY-heldout evidence only

| gate | passed |
| --- | --- |
| input_pool_loaded | True |
| all_required_domains_present | True |
| source_level_split_built | True |
| train_val_test_nonempty | True |
| each_domain_has_test_and_train_sources | True |
| test_contains_all_domains | True |
| source_file_disjoint | True |
| row_overlap_pass | True |
| no_future_or_test_leakage_constructed | True |
| old_split_reuse_boundary_recorded | True |
| no_metric_seconds_stage5c_smc_claim | True |
