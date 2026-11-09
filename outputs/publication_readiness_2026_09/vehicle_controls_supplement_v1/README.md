# Vehicle controls: two standalone evidence pages

The first page puts CV, OLS4, the vehicle incumbent and the two
matched EqMotion variants in the same table and shows the direct
paired location contrasts. The second page compares support
reranking and missing-step factorization at the same two-edit
budget, with the release counts kept separate. Both pages use
OpenDD development evidence only.

The builder reads the frozen [forecast](../opendd_vehicle_eqmotion_v1/README.md),
[support](../opendd_support_rerank_v1/README.md) and
[factorization](../opendd_pair_completion_v1/README.md) readouts,
checks their hashes, and checks that each points to its original
freeze. The [receipt](receipt.json) binds the generated PDF to the
three inputs. Two builds were byte-identical; both final pages were
rendered and visually inspected. Key numbers were checked in the
PDF text layer, and there are no author metadata or link annotations.

```sh
# Use a Python environment containing reportlab and pypdf.
python scripts/build_m3w_vehicle_supplement.py
pdftoppm -r 125 -png \
  data/stage_cvpr2027_experiments/submission_admin/vehicle_controls_supplement.pdf \
  data/stage_cvpr2027_experiments/submission_admin/vehicle_controls_preview
```

The PDF stays local. Neither existing paper nor supplementary PDF
was replaced. No upload, new training, new external readout,
calibration, deployment or new scientific claim follows from this
build. The independent Songdo comparison remains pending.
