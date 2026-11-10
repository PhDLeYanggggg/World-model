# Paper candidate with frozen Songdo confirmation

I replaced the retrospective recap in the limitations section with the
[registered 160-flight confirmation](../songdo_calibrated_20261110/README.md)
and cited [Songdo Traffic v2](https://zenodo.org/records/17924857).
The text reports the matched joint-minus-independent lower gain with
its source-bootstrap interval, then separately states the actual
release counts. It does not count CV fallback as a risk pass or
imply that the reserved DroneCrowd confirmation was opened.

Only one prose paragraph and one reference change from v169.
The title, abstract, original references, figures, tables and LLM
Usage Statement remain unchanged. The PDF is eight pages; the first
seven match pixel for pixel, and the last was inspected. A
second offline compile reproduces the same bytes. The new result is
bound to the pre-outcome calibration hash. The existing supplementary
ZIP remains unchanged; the detailed Songdo tables are in the report
above.

```sh
# Use a Python environment with pypdf and an existing Tectonic cache.
python scripts/build_m3w_songdo_candidate.py --compiler /path/to/tectonic
python -m unittest discover -s tests -p 'test_m3w_songdo_candidate.py'
```

Local PDF: `data/stage_cvpr2027_experiments/evidence_manuscript_v170_songdo/manuscript.pdf`.
Two candidate tests and 22 Songdo scoring and export tests passed.
[Artifact receipt](receipt.json). No upload or submission.
