# Frozen Harm-Head Gradient Diagnosis

Fresh autograd and inference on72/72 fit views completed in61.04s.
Both frozen predictors reproduce their original prediction hashes. No new
fit, holdout inference, threshold, deployment or independent evidence.
CREATE TRAIN files include internal held rows during transport; they were
discarded before forwarding. Cached checkpoints and preprocessing were verified.

## Same-Cohort Results

| Fixed cohort | Arm | Rows | Unknown | Easy zero-J rows | Easy underprediction | At zero J | At hard H lower clip | At hard q lower clip |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eligible | isolated_odds | 894975 | 18840 | 1271 | 3960.07525842 | 13.59004455 | 0.00000000 | 0.00000000 |
| eligible | full_range | 894975 | 18840 | 628949 | 4069.01784222 | 3260.25219444 | 64.31061746 | 3256.58611310 |
| anchor | isolated_odds | 337984 | 5363 | 1157 | 975.94597700 | 12.03051758 | 0.00000000 | 0.00000000 |
| anchor | full_range | 337984 | 5363 | 227979 | 958.88082066 | 777.20967610 | 21.64381685 | 774.54308121 |
| isolated_odds | isolated_odds | 45395 | 592 | 581 | 58.38162086 | 4.49162460 | 0.00000000 | 0.00000000 |
| isolated_odds | full_range | 45395 | 592 | 32138 | 51.29482901 | 42.19875010 | 5.86459751 | 41.18949149 |
| full_range | isolated_odds | 58707 | 1612 | 33 | 77.73770634 | 0.00000000 | 0.00000000 | 0.00000000 |
| full_range | full_range | 58707 | 1612 | 54254 | 79.54137545 | 76.59033871 | 64.31061746 | 72.92425737 |

J is the actual autograd Jacobian of predicted easy harm with respect to
both risk logits. H is all-harm; q is the easy/all-harm ratio. Clip columns
can overlap and must not be summed. Values use frozen fitting source scales;
repeated occurrences are not independent samples, ADE/FDE or probabilities.

## Interpretation Boundary

These are local derivatives at final weights. Zero J prevents direct output
correction for that input at that instant; shared weights can still move from
other rows. This does not prove irreversibility or a sole optimization cause.
Nonzero J does not imply useful parameter gradients or convergence. A realized
harm above a predicted cap does not establish an impossible conditional mean.
Original support and independent-evidence gates remain failed.

Obs8/pred12, raw stride12; detector labels, pixel/dataset-local coordinates.
No metric/seconds/physical-safety/human-gold/true3D/foundation claim.
Stage5C and SMC disabled.

## Reproduction

```bash
.venv-pytorch/bin/python -m scripts.audit_m3w_harm_jacobians check
```

Check replays immutable receipts and accounting, not another training run.
