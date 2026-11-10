# Athens and Swiss roundabouts: match movement at each forecast step

The original comparison fixes the edit count and peak departure from
constant velocity (CV). A pair can still move by different amounts
between steps. For this secondary control I retained the original
recipients and directions, then gave both edited agents in both arms
the same departure from CV at each of the twelve steps. A smaller-peak
control scales the whole path uniformly to the same new peak, without
matching the individual steps. See the [frozen protocol](protocol.md).

All 1,929 Athens matched query-seed views (four acquisition dates) and
7,431 Swiss matched query-seed views (three sites) support all three
contrasts after step matching. The other 258 Swiss views were already
outside the original matched roster. There were no degenerate common
profiles among matched views. Views are not independent samples.

| Source | Joint versus | Original ADE bound | Smaller peak | Matched steps | Integrated movement retained |
| --- | --- | ---: | ---: | ---: | ---: |
| Athens | Independent | 0.007235 | 0.006726 | 0.006018 | 84.4% |
| Athens | Additive | 0.007148 | 0.006635 | 0.005939 | 84.4% |
| Athens | Same family | 0.000337 | 0.000299 | 0.000281 | 90.9% |
| Swiss | Independent | 0.032526 | 0.030210 | 0.027476 | 85.3% |
| Swiss | Additive | 0.031373 | 0.029101 | 0.026453 | 85.3% |
| Swiss | Same family | 0.008588 | 0.007496 | 0.007041 | 88.8% |

ADE bounds are whole-roster geometric upper bounds on the magnitude of
an ADE difference, averaged with equal weight across dates or sites.
They are in dataset-local units, not pooled across datasets. Matching
every step reduced the bound even relative to the smaller-peak control,
but did not eliminate the geometric difference. No row in this table
measures an improvement in accuracy.

The [Athens](athens.json) and [Swiss](swiss.json) receipts bind the
frozen inputs and report FDE and relative-path bounds as well. The
paired scorer was rehearsed with fabricated targets and masks (162
Athens and 486 Swiss packet/contrast/variant checks), with no fabricated
effect estimates exported. No real future coordinates were read. No
roles, thresholds, primary metrics or release decisions changed. The
original two-edit peak-matched comparison and 2% release rule remain.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/freeze_m3w_external_step_budget.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_external_step_budget.py tests/test_m3w_step_budget_match.py tests/test_m3w_swiss_roundabout_readout.py tests/test_m3w_songdo_confirmation_export.py
```
