# Working Manuscript v52: Matched Transfer

Section 35 reports frozen OpenDD forecasts on the previously exposed
TUMDOT-ING campaign. The [v50 main paper](../evidence_manuscript_v50/README.md)
is unchanged. The matched result is negative against CV and a causal
hindcast. The label-aware analysis shows that better paths are available
without establishing a policy that can choose them from history alone.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v52
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v52.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

The direct and clean archive builds render the same fourteen pages.
Section 35 was visually checked. No enrollment or submission was made.

| Local artifact | SHA-256 |
|---|---|
| Supplement PDF | `7d97a7acc37e3d534a86fb55df3d589314819b26e0de394a9b7056c1054517fb` |
| Supplement source ZIP | `ba9b99d45b3ad391749fb3c56de06525c6c8aa7b38373bf654af38b9366b19f7` |

[Experiment and reproduction](../tumdot_matched_finite_transfer_v1/README.md).
