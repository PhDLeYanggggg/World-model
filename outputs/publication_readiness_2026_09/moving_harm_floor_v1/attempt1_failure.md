# Receipt Serialization Failure and Recovery

Fresh run, 5 November 2026. Training PID 9913 exited 1 after the first joint
control completed 2,000 updates. The final checkpoint was saved and exact
comparison with the preceding joint control passed. The error occurred while
exporting the fitting receipt: `TypeError: Object of type ndarray is not JSON
serializable`. The planned query counts were still a NumPy array in that receipt.
This was an export error, not a stalled optimizer or a scientific result.

The intact checkpoint is local only:
`data/stage_cvpr2027_experiments/moving_harm_floor_v1/fits/eqmotion_fold1_seed17_critic0_joint.pt.gz`.
Its SHA256 before recovery is
`e4bad40a0b2f320f87d97c43fc00697765d56fb99e638d8f48a64dae471858fd`.

The repair converts this field to a list for JSON output and verification only.
The original registration remains immutable. A separate serialization amendment
records the changed runner hash and requires every other scientific binding to
match. Models, losses, data, sampling, thresholds, original safety/support rules
and checkpoint identities are unchanged. No additional V/H readout has occurred.

Recovery uses `train --resume`. The completed first fit needs no new optimizer
updates; remaining fits retain the original budget. Recovery completion is not
asserted by this failure record. See the later execution and replay receipts.
