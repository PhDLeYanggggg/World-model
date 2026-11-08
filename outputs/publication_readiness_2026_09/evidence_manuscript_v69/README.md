# Working Supplement v69: Cross-Query Ranking

The [main paper remains v66](../evidence_manuscript_v66/README.md). The
supplement adds the fixed-recording-budget ranking ablation, the missing
label interval for swaps, and the distinction from prior decision-focused
ranking. It does not support a new online policy or change the original
risk rule or confirmation status.

```sh
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v69.py \
  --source-zip data/stage_cvpr2027_experiments/evidence_manuscript_v69/supplement_source.zip
```

Direct and clean source-archive builds each have 18 pages with identical
85 DPI page renders. The new text and citation were checked visually.
Compiled artifacts stay local.

- PDF SHA-256: `f9ee01437f09a22d03ba7207a110cd330953c6c3e8d1992404aba30a020bd9a5`
- Source ZIP SHA-256: `d1ffce9647c286e292b5f7398bd3bfc0d94714295df6c133b68fae3b35a2635b`
