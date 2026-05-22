# Final Model Selection

- case: `C`
- deployment_strategy: `strongest baseline fallback`
- checkpoint: `outputs/final_model/final_selected_checkpoint.pt`
- reason: learned correction did not clear official t+50 and hard/failure gates

Case C means the complete deployable model falls back to strongest causal baselines while reporting failure probabilities and diagnostics.
