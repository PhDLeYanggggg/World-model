# Paper candidate: missing-label score and coordinate scope

I made a separate eight-page PDF from the verified v167 source archive.
Only two prose passages change. The first distinguishes annotation
pixels, camera-detection pixels and OpenDD release coordinates. The
second distinguishes the two existing conservative utility scores:

| Score | Partial futures | Wholly missing futures | Scope |
| --- | --- | --- | --- |
| Row-level | Keep each row's mean on its available steps | Debit the maximum forecast disagreement per row | Original row-based utility; not a full-grid ADE bound |
| Stepwise | Debit the forecast disagreement at each missing waypoint | Debit every waypoint | Moving arbitrary-path audit and stepwise OpenDD and WEPDTOF contrasts |

The three 2% recording risk checks are not redefined. The paragraph
also makes clear that 14 sources is the primary Moving bootstrap
count, not the count for every dataset. The synthetic scoring test
calls both original implementations: the same forecasts and labels
produce a row-level lower gain of -50% and a stepwise lower gain of
-300% in the deliberately small example. Treating them as one metric
would not reproduce the result.

The title, abstract, tables, figures, bibliography, and LLM Usage
Statement are unchanged. The new PDF was rendered at 90 DPI and the
changed pages and final disclosure page were visually checked. The
anonymous eight-page build reproduced byte for byte. The earlier
v167 PDF and supplementary bundle remain unchanged and local.

```sh
# Use a Python environment with pypdf and an existing Tectonic cache.
python scripts/build_m3w_metric_scope_candidate.py --compiler /path/to/tectonic
python -m unittest discover -s tests -p 'test_m3w_metric_scope_candidate.py'
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_manuscript_score_scope.py tests/test_m3w_wepdtof_frozen_capacity.py tests/test_m3w_moving_peak_certificate.py
```

The candidate is at `data/stage_cvpr2027_experiments/evidence_manuscript_v168_metric_scope/manuscript.pdf`.
Fourteen focused tests passed. [Artifact checksums](receipt.json).
No new outcomes, edits to v167, or portal upload.
