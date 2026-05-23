# Stage 24 SDD I/O Profile

- single shard open time: `0.0002s`
- sequential read time: `0.0054s`
- random read source seconds: `{'100': 0.014811917000001174, '1000': 0.14940333300000042, '10000': 1.4940333300000042}`
- random read fast cache seconds: `{'10000': 0.11607545800000096}`
- bottleneck: `compressed NPZ plus repeated agent/frame lookup`
- recommended cache: `per-video uncompressed .npy memmap arrays + JSON track/frame indexes`
- must build fast cache first: `False`
