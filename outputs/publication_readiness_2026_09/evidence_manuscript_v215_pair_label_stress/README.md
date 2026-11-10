# Manuscript v215: where labels are missing matters

The new paragraph compares early and late missing steps at the same
75% coverage. On the 3,711 complete OpenDD queries, the observed
EqMotion neighbor gain is +0.035 when the first three steps are hidden,
-0.470 when the last three are hidden, and -0.209 with all labels.
The original forecasts and external results are unchanged.

The anonymous main PDF, source ZIP and separate figure
`equal_coverage.pdf` are local in
`data/stage_cvpr2027_experiments/evidence_manuscript_v215_pair_label_stress/`.
The title, abstract and LLM Usage Statement are unchanged.
The paired `review_supplement_v49.zip` in `submission_admin/`
contains the figure and a standard-library replay of the paired
location statistics. It preserves v48 evidence and the v44 reader
PDF. Nothing was uploaded or submitted.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_pair_label_stress.py
python scripts/build_m3w_pair_label_stress_candidate.py --compiler "$TECTONIC"
```

[Experiment](../opendd_pair_label_stress_v1/README.md) |
[Build receipt](receipt.json) | [Replay receipt](review_receipt.json) |
[Chinese guide](../../../reproducibility/M3W_PAPER_REPLAY_ZH.md).
