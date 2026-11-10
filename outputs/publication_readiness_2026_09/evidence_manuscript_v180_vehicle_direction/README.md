# v180: fixed-recipient vehicle direction comparison

This candidate keeps the v179 title, abstract, author disclosure and
external results. It adds one paragraph to the joint comparison:
493 OpenDD vehicle development recordings, with separate forced
two-rotation and optional single-rotation controls. The result does
not support replacing the original direction rule.

From the project root:

```sh
.venv-pytorch/bin/python scripts/build_m3w_vehicle_direction_candidate.py \
  --compiler "/Applications/ChatGPT.app/Contents/Resources/tectonic/tectonic"
```

The build uses the local v179 source and the two direction readouts.
It writes `data/stage_cvpr2027_experiments/evidence_manuscript_v180_vehicle_direction/manuscript.pdf`
and `main_source.zip`. The [receipt](receipt.json) binds the inputs and
PDF, and records the six-page compile. The changed pages were visually
checked. The [v10 supplement](../vehicle_direction_review_supplement_v1/README.md)
adds anonymous replay. Nothing was uploaded or marked submission-ready.
