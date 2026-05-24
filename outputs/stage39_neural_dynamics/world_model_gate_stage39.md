# Stage39 Gates

- gates passed: `11 / 13`
- verdict: `stage39_neural_dynamics_diagnostic_keep_stage37`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| Gate1 Stage37 floor frozen | True | 3ee52d057ab49e3952b076d59657a0a2a7f939d74d99a3f3cea8cf37a55673af |
| Gate2 neural dataset built no leakage | True | {'train': {'split': 'train', 'source': 'cached_verified', 'rows': 24000, 'path': 'data/stage39_neural_dynamics/neural_dataset_train.npz'}, 'val': {'split': 'val', 'source': 'cached_verified', 'rows': 8000, 'path': 'data/stage39_neural_dynamics/neural_dataset_val.npz'}, 'test': {'split': 'test', 'source': 'cached_verified', 'rows': 16000, 'path': 'data/stage39_neural_dynamics/neural_dataset_test.npz'}} |
| Gate3 Transformer trained | True | {'val_loss': 38.90000534057617, 'epoch': 1, 'train_loss': 0.18872265073847264} |
| Gate4 JEPA trained and non-collapse checked | True | {'val_loss': 79.89398956298828, 'epoch': 2, 'train_loss': 0.0008967651470385968, 'latent_variance': 0.06202657148241997} |
| Gate5 Hybrid trained | True | {'val_loss': 17.962100982666016, 'epoch': 4, 'train_loss': 0.01619455030069072} |
| Gate6 neural_with_fallback beats Stage37 on all/t50/hard at least one | False | {'rows': 16000, 'all_improvement': 0.13200527968500975, 't10_improvement': 0.3025072845476737, 't25_improvement': 0.0, 't50_improvement': 0.08301228991324938, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1523895057047563, 'easy_degradation': 0.0006815189764808327, 'harm_over_fallback': -0.13887056636996567} |
| Gate7 easy degradation <=2 | True | 0.0006815189764808327 |
| Gate8 SDD safety pass | True | preserved_by_not_deploying_neural_on_sdd |
| Gate9 external held-out domains repaired or honest blocker | True | {'UCY': 'available_heldout_test', 'ETH_UCY': 'not_run_blocker: available rows are train-only under frozen Stage37 split; rebuilding held-out test would invalidate frozen policy/test protocol', 'TrajNet': 'not_run_blocker: train/val rows exist but no frozen held-out test split; requires Stage40 split rebuild and retuning on val only', 'OpenTraj_mixed': 'not_run_blocker: mixed test currently UCY; non-UCY held-out requires new split'} |
| Gate10 t100 diagnostic honest | True | 0.0 |
| Gate11 world dynamics candidate gate | False | keep_stage37_selector |
| Gate12 Stage5C false | True | Stage5C not executed |
| Gate13 SMC false | True | SMC not enabled |
