# Manuscript v216: a single reader appendix

The introduction now separates feasibility, selection and frozen
transfer. The title, abstract, disclosure and numerical results
stay unchanged. The reader appendix retains the four original
pages and adds the training and scoring recipe, the exact-spend
and listwise tables, the relative-trajectory results and the
equal-coverage figure. It explains why the broader 330-pass
hindsight comparison is not the fixed-count, fixed-motion
210-pass comparison, and distinguishes the two uniform controls.

The local handoff is
`data/stage_cvpr2027_experiments/submission_admin/local_handoff_v216/`.
Its SHA-256 manifest binds `paper.pdf`, `supplement.pdf`, and
`replay.zip`. Both PDFs have eight pages. The supplement is
readable without running code or following external links.
The ZIP includes that same supplement and standard-library
replay of the tables. It does not include weights or raw data.
Nothing was uploaded or submitted.

```sh
python scripts/build_m3w_integrated_reader_candidate.py --compiler "$TECTONIC"
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_integrated_reader.py
```

The builder needs the local v215 handoff artifacts, the two
backbone registrations, the original exact-count receipt,
pypdf, ReportLab and cached Tectonic files. The reviewer can
extract `replay.zip` and run `python -I -B -S verify.py` offline.
The check verifies aggregate evidence, not training or inference.

[Build receipt](receipt.json) | [Reader receipt](review_receipt.json) |
[Chinese guide](../../../reproducibility/M3W_PAPER_REPLAY_ZH.md).
