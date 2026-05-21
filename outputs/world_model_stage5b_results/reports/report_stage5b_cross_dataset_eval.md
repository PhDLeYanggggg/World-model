# Stage 5B Cross-Dataset Evaluation

This quick run performs a diagnostic cross-dataset matrix over actual converted datasets. It does not claim true leave-one-dataset-out learned transfer because the Stage 5B deterministic model is a dataset-specific linear residual head.

| train | test | horizon | strongest baseline FDE | learned FDE | note |
| --- | --- | ---: | ---: | ---: | --- |
| eth_ucy | eth_ucy | 10 | 0.713643 | 0.575894 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| eth_ucy | tgsim | 100 | 6.062032 | 7.266007 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| eth_ucy | tgsim_i90 | 100 | 10.327657 | 10.618509 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| eth_ucy | trajnet | 10 | 1.434586 | 1.990041 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| tgsim | eth_ucy | 10 | 0.713643 | 0.575894 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| tgsim | tgsim | 100 | 6.062032 | 7.266007 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| tgsim | tgsim_i90 | 100 | 10.327657 | 10.618509 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| tgsim | trajnet | 10 | 1.434586 | 1.990041 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| tgsim_i90 | eth_ucy | 10 | 0.713643 | 0.575894 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| tgsim_i90 | tgsim | 100 | 6.062032 | 7.266007 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| tgsim_i90 | tgsim_i90 | 100 | 10.327657 | 10.618509 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| tgsim_i90 | trajnet | 10 | 1.434586 | 1.990041 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| trajnet | eth_ucy | 10 | 0.713643 | 0.575894 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| trajnet | tgsim | 100 | 6.062032 | 7.266007 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| trajnet | tgsim_i90 | 100 | 10.327657 | 10.618509 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
| trajnet | trajnet | 10 | 1.434586 | 1.990041 | true leave-one-dataset-out training is not available in this quick deterministic linear residual run |
