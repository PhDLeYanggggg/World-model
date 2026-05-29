# Stage43-C Protected Latent-State Training

- source: `fresh_stage43_c_protected_latent_state_small`
- result source: `fresh_run`
- mode: `quick`
- checkpoint: `outputs/stage43_latent_state/checkpoints/stage43_protected_latent_small.pt`
- checkpoint committed: `False`
- runtime: `{'python': '3.11.1', 'machine': 'arm64', 'torch_version': '2.12.0', 'torch_threads': 4, 'torch_interop_threads': 1, 'device': 'cpu', 'num_workers': 0}`
- data rows: `{'train': 4000, 'val': 2000, 'test': 2000}`

This is a protected latent-state head: z_t is learned from causal inputs and z_t -> z_{t+h} is trained against label-only future latent targets. It is not Stage5C latent generative rollout and does not enable SMC.
