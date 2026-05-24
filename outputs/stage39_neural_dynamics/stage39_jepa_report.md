# Stage39 JEPA Report

- source: `fresh_run`
- JEPA is representation training only; no pixel reconstruction, no latent rollout.
- report: `{'source': 'fresh_run', 'runtime': {'python': '/Users/yangyue/Downloads/World/.venv-pytorch/bin/python', 'machine': 'arm64', 'torch_threads': 4, 'num_workers': 0}, 'checkpoint': 'outputs/stage39_neural_dynamics/checkpoints/jepa_best.pt', 'heartbeat': 'outputs/stage39_neural_dynamics/jepa_heartbeat.json', 'best': {'val_loss': 79.89398956298828, 'epoch': 2, 'train_loss': 0.0008967651470385968, 'latent_variance': 0.06202657148241997}, 'non_collapse': True, 'downstream_lift': {'source': 'fresh_run', 'failure_auroc_base': 0.9880898558164535, 'failure_auroc_with_jepa': 0.5595315549772284, 'failure_auroc_lift': -0.4285583008392251, 'failure_auprc_base': 0.9858232078403314, 'failure_auprc_with_jepa': 0.454838276741377}}`
