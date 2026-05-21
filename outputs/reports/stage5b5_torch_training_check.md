# Stage 5B.5 Torch Training Check

Torch backend completed without the previous OpenMP/SHM hang. No Stage 5B.5 training process remains running.

| mode | checkpoint | final_loss | target wins | note |
| --- | --- | ---: | ---: | --- |
| direct_multi_horizon | outputs/checkpoints/stage5b5/temporal_interaction_direct_multi_horizon.pt | 0.832052 | 0/4 | runs, but deterministic gate still weak |
| recurrent_rollout | outputs/checkpoints/stage5b5/temporal_interaction_recurrent_rollout.pt | 0.000777 | 0/4 | runs, but deterministic gate still weak |
| hybrid | outputs/checkpoints/stage5b5/temporal_interaction_hybrid.pt | 0.642651 | 1/4 | runs, but deterministic gate still weak |
