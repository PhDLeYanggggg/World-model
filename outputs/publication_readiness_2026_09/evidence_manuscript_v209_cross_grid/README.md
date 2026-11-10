# Manuscript v209: separate risk from utility on both external cohorts

The WEPDTOF paragraph now reports that three of the four full-grid
joint risk passes cannot beat both motion references, even under
favorable missing positions. Songdo reports the difference in
guaranteed easy support. The title, abstract, references, disclosure,
prediction policies and release decisions are unchanged.

The seven-page anonymous manuscript and source ZIP are local in
`data/stage_cvpr2027_experiments/evidence_manuscript_v209_cross_grid/`.
The paired four-page supplement `review_supplement_v43.pdf` and replay
`review_supplement_v43.zip` are in `submission_admin/` alongside the
previous frozen versions. No artifact has been submitted or uploaded.

```sh
python scripts/build_m3w_full_grid_candidate.py --compiler "$TECTONIC"
python scripts/build_m3w_cross_grid_supplement.py
```

The builders use the existing verified artifacts, `pypdf`, `reportlab`
and cached Tectonic inputs. `python verify.py` in the unpacked ZIP
replays the archived counts with the standard library. See the
[Chinese guide](../evidence_manuscript_v202_focused_evidence/reproduction_zh.md)
for the broader local and CREATE experiments.

[WEPDTOF analysis](../wepdtof_full_grid_risk_v1/README.md) |
[Build receipt](receipt.json) | [Replay receipt](review_receipt.json).
