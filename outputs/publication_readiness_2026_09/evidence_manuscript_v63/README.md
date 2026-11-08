# Working Decision Supplement v63

The main manuscript stays at v50. This supplement adds the same-count
swap experiment and its exact fixed-budget hindsight audit. This is a
development comparison, not evidence of independent effectiveness.
The DroneCrowd 15/30 split remains sealed for confirmation under the
original rule. No deployment change follows.

```sh
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v63.py \
  --source-zip data/stage_cvpr2027_experiments/evidence_manuscript_v63/supplement_source.zip
```

The builder verifies lineage and binds numerical macros to the registered
readout. Local direct and archive-source builds both have 17 pages and
identical 85 DPI page renders. The final two pages were visually checked.
The generated archive and PDF remain local.

- PDF SHA-256: `c38157a9edcfe1bb8d4eb29f25dbb5d8927a6e9a9052300d543047d2202f510e`
- Source ZIP SHA-256: `385643dd9a8313080084017513823e5f53cdb064f8b8310b25c9e02f744c45fe`
