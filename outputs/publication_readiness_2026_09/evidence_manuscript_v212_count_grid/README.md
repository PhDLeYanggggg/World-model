# Manuscript v212: optimize the complete-grid test at fixed spend

The main paper now reports 212 feasible OpenDD count-and-motion views
under the original rule and 210 that also meet the full-grid conditions.
Exact enumeration resolves the two cases the original solver left
uncertain. It does not change the predictions, edit budgets, release rule
or external results. The method explains how the complete-grid excesses
become linear constraints on the same prefix choices.

The seven-page anonymous main PDF and source ZIP are local in
`data/stage_cvpr2027_experiments/evidence_manuscript_v212_count_grid/`.
The reader PDF is unchanged (`review_supplement_v44.pdf`); the paired
`review_supplement_v46.zip` in `submission_admin/` replays the new counts,
three paired location intervals, and inherited results. Nothing was
uploaded or submitted.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_opendd_full_grid_count_motion.py
python scripts/build_m3w_full_grid_count_candidate.py --compiler "$TECTONIC"
```

The first command needs the verified local OpenDD assets. The second uses
the aggregate readout, the verified v211 artifacts, `pypdf`, `reportlab`,
NumPy, and cached Tectonic inputs. `python verify.py` in the unpacked
anonymous ZIP runs using the standard library.

[Analysis](../opendd_full_grid_count_motion_v1/README.md) |
[Build receipt](receipt.json) | [Replay receipt](review_receipt.json) |
[Chinese guide](../../../reproducibility/M3W_PAPER_REPLAY_ZH.md).
