# Stage30 D External Topdown Validation

- source: `fresh_run` for path check and conversion attempt.
- Non-SDD conversion is diagnostic unless scale, scene, and latent alignment are validated.
- found paths: `['external_data/OpenTraj', 'external_data/StanfordDroneDataset', 'external_data']`
- converted files: `11`
- rows: `{'train': 52808, 'val': 3841, 'test': 808}`
- conversion status: `converted_diagnostic_non_sdd`
- transfer eval: `{'status': 'not_run', 'reason': 'Full M3W-LAS all_latent transfer needs external latent cache and scale calibration; converted base feature store is diagnostic only.'}`
- no leakage: `{'split_by_file': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False}`

## Strongest Baselines
- train: `constant_velocity_causal_fd`
- val: `constant_velocity_causal_fd`
- test: `constant_velocity_causal_fd`
