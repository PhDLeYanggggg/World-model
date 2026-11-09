# Who to edit and which path to use

I separated recipient ranking from the choice of a replacement path.
From two earlier analytic witnesses per training recording, I built
237 distinct, independently verified recording plans. Across the
three train roles, 53,445 agent windows are selected in some but not
all plans. Their selection frequency is a soft label for this
particular family, not a probability over every feasible plan.
The single-plan and soft-label models have the same architecture,
sampling and update count. A separate action model learns from
paired alternatives on the same recipient and may not move farther
from CV than the frozen cost action on that recipient.

## Results at the same edit count

There were 27 native PyTorch fits, 32,400 optimizer updates, three
seeds and three source-role rotations. Each matched arm edited
45,698 rows with the same count in each query. The earlier frozen
policy, listed separately, keeps its own budget.

| Recipient rank / path rule | Edits | Original risk passes / 72 | Worst known signed easy change |
|---|---:|---:|---:|
| Earlier frozen policy | 54,748 | 14 | +61.90% |
| Cost / frozen cost path | 45,698 | 0 | +291.50% |
| Cost / paired path | 45,698 | 1 | +84.37% |
| Single plan / frozen cost path | 45,698 | 0 | +256.82% |
| Single plan / paired path | 45,698 | 1 | +76.96% |
| Soft plans / frozen cost path | 45,698 | 0 | +348.90% |
| Soft plans / paired path | 45,698 | 3 | +77.31% |

At the same cost path rule, single-plan ranking loses 1.238 paired
lower pp versus cost-only (-2.105 to -0.524). Soft ranking loses
another 1.529 lower pp versus single-plan (-2.829 to -0.529).
With recipients fixed to cost, paired path ranking gains 4.916 pp
on observed steps but does not show a conservative gain
(-21.972 pp, -61.255 to +4.968). The soft/paired combination loses
3.394 paired lower pp (-6.252 to -0.956) against the earlier
frozen policy at its own budget. The combined conservative
pair-distance contrast versus cost-only is -20.975 pixels
(-52.252 to -0.007). Intervals use 3,000 paired source-group
draws after averaging seeds. **Calibration: 0/9.**

## Does changing the path add anything beyond shrinking it?

The learned path swaps 33,018 of 45,698 actions on cost-ranked
recipients and uses 30.68% of the old aggregate peak displacement.
I shrank each old path toward CV to exactly the new path's
per-recipient peak. This simple control already gives +4.662 pp
on observed steps over the unchanged cost path. At equal peak
displacement the learned path adds +0.254 observed pp (+0.004 to
+0.564), but loses 1.133 paired lower pp (-2.318 to -0.201).
The matched shrinkage control's worst signed easy change is
+214.63%, compared to +84.37% for the learned path. Shrinkage
explains most of the observed-accuracy change, not all of the
easy-risk change. No strategy qualified.

The result argues against putting another ranking loss on top of
these labels without a stronger control. The next comparison
should keep the stronger frozen recipient policy fixed and make
same-recipient shrinkage the reference. Recipient eligibility
should be learned from measured support and conservative utility,
not just membership frequency in one solver-generated family.
[Decision-focused learning to rank](https://proceedings.mlr.press/v162/mandi22a.html)
already develops rankings over feasible solutions. The contribution
tested here is the separate attribution of recipient and path
effects under fixed counts, risk rules and displacement, not a
claim to invent multi-solution learning. [Decision-Point Guided Safe Policy Improvement](https://proceedings.mlr.press/v258/sharma25a.html)
motivates looking at support for departures, but its batch-RL
guarantee does not carry over to this forecasting contract.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/build_m3w_cv_recipient_teachers.py
.venv-pytorch/bin/python scripts/train_m3w_cv_recipient_path.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_recipient_radius.py
.venv-pytorch/bin/python scripts/build_m3w_cv_recipient_path_replay.py
```

Teacher construction, training and radius controls are `fresh_run`.
The frozen bank, controls and no-update checkpoint replay are
`cached_verified`. Remove `--replay` to train or resume after
checking the running process and heartbeat. The clean NumPy
replay checks scalar policy tables, not training or pair geometry.
All results are native 8 observed / 12 future steps in
dataset-local pixels. No deployment, new A outcomes or
independent confirmation; 30 DroneCrowd confirmation groups stay sealed.

[Protocol](protocol.md) | [Main readout](readout.json) |
[Equal-radius control](radius_control.json) |
[Manuscript v136](../evidence_manuscript_v136/README.md)
