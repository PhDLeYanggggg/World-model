# Reader supplement v42 for manuscript v208

This version adds the full-grid selected-easy calculation to page 3.
It keeps the main PDF, the release rule, and the other supplement pages
unchanged. The table separates guaranteed easy reference support from
passing the numerical checks and from an actual release. See the
[analysis and derivation](../songdo_full_grid_easy_risk_v1/README.md).

The local submission folder contains `review_supplement_v42.pdf` and
`review_supplement_v42.zip`. The ZIP includes an offline aggregate
replay and its v41 parent, without source trajectories or checkpoints.
The [receipt](receipt.json) binds both files to the unchanged main PDF.
Neither has been uploaded.

```sh
python scripts/build_m3w_reader_supplement.py --full-grid-easy
```

The builder uses `pypdf`, `reportlab`, and the existing verified inputs.
The archive's `python verify.py` uses only the standard library. The
[Chinese guide](../evidence_manuscript_v202_focused_evidence/reproduction_zh.md)
covers the broader experiments and CREATE recovery.
