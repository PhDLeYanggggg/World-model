# Manuscript v210: how much of the selection gap survives missing futures?

The paper now describes the full-grid risk test in the method and
reports the OpenDD one-exchange result alongside the matched-budget
experiment. The original 181-view hindsight result becomes 175 when
also requiring full-grid risk. On the aligned ranker's fixed switch
mask, hindsight finds 98 passes, the ranker finds 44, and uniform
selection expects 46.73. This is development evidence, not a new
confirmation or a change to the release rule.

The seven-page manuscript and source ZIP are local in
`data/stage_cvpr2027_experiments/evidence_manuscript_v210_exchange_grid/`.
The four-page `review_supplement_v44.pdf` and the paired replay ZIP
are in `submission_admin/`. The ZIP replays counts and paired location
intervals offline. No raw trajectories or checkpoints are
included, and nothing has been submitted or uploaded.

```sh
python scripts/build_m3w_full_grid_exchange_candidate.py --compiler "$TECTONIC"
python scripts/build_m3w_cross_grid_supplement.py --exchange
```

The builders use `pypdf`, `reportlab`, NumPy, and the cached Tectonic
inputs. `python verify.py` in the unpacked ZIP uses the standard library.
See the [Chinese replay guide](../evidence_manuscript_v202_focused_evidence/reproduction_zh.md)
for local training and CREATE recovery.

[OpenDD analysis](../opendd_full_grid_exchange_v1/README.md) |
[Build receipt](receipt.json) | [Supplement receipt](review_receipt.json).
