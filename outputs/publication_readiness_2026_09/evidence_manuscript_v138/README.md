# Evidence manuscript v138

The new supplement section reports the recipient exchange
experiment at matched edit count and peak displacement, and the
query-local and recording-wide capacity bounds. The result
remains development-only; no candidate qualified for confirmation.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v138
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v138.py main --source-zip "$out/main_source.zip"
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v138.py supplement --source-zip "$out/supplement_source.zip"
(cd "$out/build" && tectonic -X compile manuscript.tex)
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

| Local artifact | SHA-256 |
|---|---|
| `build/manuscript.pdf` (8 pages) | `cea246d58565b5e45546010bd21c0f0d481a3cd189f868c2d6c45402d04b1a7c` |
| `supplement/supplement.pdf` (41 pages) | `6fb64808b40a0931a9c73bf381c088c5bf6fa26b323e31748e65139035295e68` |
| `main_source.zip` | `913d343dcfbffc4c93bdbbcdb99ee6600999e446ce07dad970006e389639dc32` |
| `supplement_source.zip` | `47820bc89e038ca32fef449d40f4d2415b4aef02d0d03bea9e168b58eefec547` |
| `../moving_cv_matched_recipient_exchange_v1/anonymous_matched_exchange.zip` | `c2b43c7ef268aec10e811e70d1cea37bab6d81332556d020709e929f4cf8b09b` |

All 49 rendered pages match clean source-archive builds at 85 DPI
with the separately supplied style. The changed pages were
inspected. The anonymous archive verifies the scalar policy table
after clean extraction. It is not a replay of the training or
path-level bounds. PDFs and archives remain local.

[Exchange experiment](../moving_cv_matched_recipient_exchange_v1/README.md) |
[Chinese reproduction guide](../../../reproducibility/M3W_CURRENT_ZH.md).
