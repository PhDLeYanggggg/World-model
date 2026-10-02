# Registered Advancement Screen

This is a source-development advance-to-transfer screen, not a deployment gate
or a generalization guarantee. The original 2% budget is unchanged.

| Requirement | Result |
|---|---|
| Quality MSE CI upper <0 vs original | Pass: -0.011377 |
| Quality MSE CI upper <0 vs placebo | Pass: -0.016930 |
| Full utility CI lower >0 vs original | Pass: +0.126543% |
| Matched utility CI lower >0 vs original | Pass: +0.020429% |
| Full utility CI lower >0 vs placebo | Pass: +0.080971% |
| Matched utility CI lower >0 vs placebo | Pass: +0.001628% |
| Complete support no lower than original | Fail: 21 vs33 |
| Easy-risk upper violations no more than original | Fail: 41 vs7 |
| Worst easy-risk upper no larger than original | Fail: 10.308127 vs0.054058 |

**advance_to_transfer = false.** Six favorable predictive contrasts cannot vote
away three failed safety conditions. No transfer was executed, independent roles
remain closed, deployment unchanged. Stage5C execution=false; SMC=false.

Separate engineering checks pass: registration/source hashes, all144 exact refits,
serialization and inference replay, original action anchors, remote checkpoint
hashes, 13,248 scalar/parent checks and 440 independently reduced summary fields.
There are 19 passing scoped tests (15 unchanged method/readout tests reused,
four new diagnostic tests). These checks do not turn the research gate green.
