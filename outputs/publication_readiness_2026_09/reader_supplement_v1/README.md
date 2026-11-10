# Reader supplement for manuscript v208

I paired the unchanged v208 manuscript with a four-page anonymous
supplement. It explains the dataset roles and matched comparisons,
shows the development and external evidence, and closes with three
Songdo cases. The [full-grid check](../songdo_full_grid_all_risk_v1/README.md)
is a separate sensitivity analysis. The original release rule and
conclusions are unchanged.

The local submission folder contains `review_supplement_v41.pdf` and
`review_supplement_v41.zip`. The ZIP contains the PDF, an aggregate replay
of the new risk table, and the v40 replay package. It contains neither
raw trajectory labels nor checkpoints. Nothing was uploaded. The
[receipt](receipt.json) binds both files to the exact main PDF.

With `reportlab`, `pypdf`, and the verified local artifacts available:

```sh
python scripts/build_m3w_reader_supplement.py
```

Run `python verify.py` in the unpacked ZIP for offline aggregate replay.
See the [Chinese reproduction guide](../evidence_manuscript_v202_focused_evidence/reproduction_zh.md)
for the broader experiment setup.
