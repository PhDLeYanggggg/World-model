# Stage43-G Source-Level Protected Latent Training

- source: `fresh_stage43_g_source_level_protected_latent`
- mode: `full`
- checkpoint: `outputs/stage43_latent_state/checkpoints/stage43_source_level_latent_full.pt`
- checkpoint committed: `False`
- data rows: `{'train': 146809, 'val': 101446, 'test': 89736}`
- runtime: `{'python': '3.11.1', 'machine': 'arm64', 'torch_version': '2.12.0', 'torch_threads': 4, 'torch_interop_threads': 1, 'device': 'cpu', 'num_workers': 0}`

This training uses the Stage43-F source-file-level split. Future endpoint/waypoint labels remain loss/eval only.
