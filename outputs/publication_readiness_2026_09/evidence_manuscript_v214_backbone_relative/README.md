# Manuscript v214: compare relative motion on the same vehicles

The OpenDD table now reports relative ADE on pair-steps with both
targets labeled. The added text distinguishes the neighbor ablations:
observed relative-ADE gain is -0.825 for EqMotion and +0.761 for
Social-STGCNN. The full-grid envelopes include zero in both cases.
The matched rows in the table use neighbors; the ego-only EqMotion
result is stated separately. No external result, forecast weight,
threshold or release decision changed.

The eight-page anonymous main PDF and source ZIP are local in
`data/stage_cvpr2027_experiments/evidence_manuscript_v214_backbone_relative/`.
The title, abstract and LLM Usage Statement are unchanged. The paired
`review_supplement_v48.zip` in `submission_admin/` replays support
counts, relative errors and paired location intervals. It retains the
v44 reader PDF and the evidence from v47. Nothing was uploaded or submitted.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_backbone_relative.py
python scripts/build_m3w_backbone_relative_candidate.py --compiler "$TECTONIC"
```

The audit needs the frozen OpenDD assets. The builder uses the
readout, verified v213 artifacts, NumPy, pypdf, ReportLab and cached
Tectonic files. The anonymous verifier only needs Python's standard
library. It replays aggregate arithmetic, not model inference.

[Experiment](../opendd_backbone_relative_v1/README.md) |
[Build receipt](receipt.json) | [Replay receipt](review_receipt.json) |
[Chinese guide](../../../reproducibility/M3W_PAPER_REPLAY_ZH.md).
