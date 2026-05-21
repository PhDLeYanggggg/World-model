# Stage 5B.5 Cross-Dataset Evaluation

| train_dataset | test_dataset | target_horizon | diagnostic_learned_fde | strongest_baseline_fde | improvement | note |
| --- | --- | --- | --- | --- | --- | --- |
| eth_ucy | eth_ucy | 10 | 0.713643 | 0.713643 | 0.0 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| eth_ucy | tgsim | 100 | 6.062032 | 6.062032 | 0.0 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| eth_ucy | tgsim_i90 | 100 | 9.411212 | 10.327657 | 0.088737 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| eth_ucy | trajnet | 10 | 1.488781 | 1.434586 | -0.037778 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| tgsim | eth_ucy | 10 | 0.713643 | 0.713643 | 0.0 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| tgsim | tgsim | 100 | 6.062032 | 6.062032 | 0.0 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| tgsim | tgsim_i90 | 100 | 9.411212 | 10.327657 | 0.088737 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| tgsim | trajnet | 10 | 1.488781 | 1.434586 | -0.037778 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| tgsim_i90 | eth_ucy | 10 | 0.713643 | 0.713643 | 0.0 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| tgsim_i90 | tgsim | 100 | 6.062032 | 6.062032 | 0.0 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| tgsim_i90 | tgsim_i90 | 100 | 9.411212 | 10.327657 | 0.088737 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| tgsim_i90 | trajnet | 10 | 1.488781 | 1.434586 | -0.037778 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| trajnet | eth_ucy | 10 | 0.713643 | 0.713643 | 0.0 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| trajnet | tgsim | 100 | 6.062032 | 6.062032 | 0.0 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| trajnet | tgsim_i90 | 100 | 9.411212 | 10.327657 | 0.088737 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
| trajnet | trajnet | 10 | 1.488781 | 1.434586 | -0.037778 | diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out |
