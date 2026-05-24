# Stage28 M3W Latent Cache Report

- 当前不是 true 3D，也不是 foundation world model；SDD 仍是 pixel-space raw-frame benchmark。
- Latent cache is aligned to Stage26 causal feature rows and uses frozen M3W checkpoints only.
- No future endpoint, central velocity, or test endpoint goals are used as input.

- cache dir: `data/stage28_m3w_latent_cache`
- elapsed seconds: `20.975`
- runtime: `{'platform_machine': 'arm64', 'torch_threads_env': '4', 'torch_interop_threads_env': '2', 'num_workers': 0}`

| split | rows | path |
| --- | ---: | --- |
| train | 40000 | `data/stage28_m3w_latent_cache/train.npz` |
| val | 20000 | `data/stage28_m3w_latent_cache/val.npz` |
| test | 100000 | `data/stage28_m3w_latent_cache/test.npz` |
