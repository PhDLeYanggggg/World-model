# Manuscript v211: the full-grid test and the selection gap

The method now states the complete-grid risk test as three inequalities
and explains why optional easy membership changes the worst case. A
new OpenDD figure compares original and complete-grid passes at fixed
edit budgets, and shows the seven paired location effects. The main
text includes the equal-training pointwise and listwise results. The
evaluation rule, frozen decisions, title, abstract and disclosure are
unchanged. No new independent targets were accessed for this revision.

The seven-page main PDF and source ZIP are local in
`data/stage_cvpr2027_experiments/evidence_manuscript_v211_exchange_exposition/`.
The same four-page reader PDF (`review_supplement_v44.pdf`) is paired
with an expanded anonymous replay ZIP (`review_supplement_v45.zip`)
under `submission_admin/`. The ZIP reproduces the pointwise/listwise
counts and five paired location intervals with the standard library.
No trajectories or checkpoints are included. Nothing was uploaded.

```sh
.venv-pytorch/bin/python scripts/plot_m3w_exchange_gap.py
python scripts/build_m3w_exchange_exposition_candidate.py --compiler "$TECTONIC"
python scripts/build_m3w_exchange_exposition_replay.py
```

The figure uses Matplotlib. The paper and replay builds use `pypdf`,
`reportlab`, NumPy and the existing cached Tectonic inputs. The
[Chinese guide](../../../reproducibility/M3W_PAPER_REPLAY_ZH.md)
describes local training, CREATE and offline reproduction.
[Build receipt](receipt.json) | [Replay receipt](review_receipt.json).
