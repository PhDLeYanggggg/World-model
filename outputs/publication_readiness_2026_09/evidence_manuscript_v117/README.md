# Evidence Supplement v117

This adds the [Hannover shared-displacement result](../hannover_shared_radius_v1/README.md),
its upper-bound derivation, and the finite-bank comparison. It changes
neither the [v116 main manuscript](../evidence_manuscript_v116/README.md)
nor the independent-evaluation claim. The numerical table is generated
from the registered readout.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v117
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v117.py --source-zip "$out/supplement_source.zip"
(cd "$out/supplement" && tectonic -X compile supplement.tex)
```

All 32 pages matched the clean source-archive render at 85 DPI with
the same separately supplied CVPR style. The new section on page 32
was visually checked. No submission or new independent confirmation.

| Local artifact | SHA-256 |
| --- | --- |
| `supplement/supplement.pdf` | `cf024f47998842fbf77f07d85db18201b58c0f60257d2577ea872c6b277babcc` |
| `supplement_source.zip` | `49e398436457a2abdfeed46f43d72f04c2d5a92d1f10ebb45d1400b69db03fd6` |
