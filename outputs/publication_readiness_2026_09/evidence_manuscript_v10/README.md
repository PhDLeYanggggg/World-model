# Native Moving-Drone Working Manuscript

This version separates the native MovingDroneCrowd++ observation8/prediction12
task at stride1 from the earlier European stride12 study and historical SDD
raw-frame studies. Their absolute errors and gains are not pooled.

The draft integrates completed support, prefix-loss and past-image cost-head
experiments, including their negative registered contrasts and failed original
recording-risk checks. The new twelve-model past-image actor matrix is still
training. Its outcomes are not in this version. No new outcome scoring is done
by the manuscript builder.

This is a working evidence draft, **not submission-ready**. Exact replay and a
compiled PDF do not establish the method contribution or independent evidence.
The original2%/support/NA requirements remain; identity certification is not
introduced. Stage5C execution and SMC remain off.

## Files And Reproduction

- `manuscript.template.md`: editable English narrative and formulas.
- `manuscript.md` and `manuscript.tex`: generated from public scalar receipts.
- `source_bindings.json`: pinned evidence and generated-source hashes.
- `build_verification.json`: current PDF and visual-check record.

```sh
.venv-pytorch/bin/python scripts/build_m3w_native_paper_v10.py
python3 /Users/yangyue/.codex/plugins/cache/openai-bundled/latex/0.2.8/scripts/compile_latex.py \
  data/stage_cvpr2027_experiments/evidence_manuscript_v10/paper_build/main.tex \
  --compiler tectonic --json
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_manuscript.py
```

The local generated PDF and scalar-only figures are under
`data/stage_cvpr2027_experiments/evidence_manuscript_v10/paper_build/`.
No dataset images, third-party style file, raw data, weights or caches are
committed. A standalone anonymous reconstruction package is still outstanding.
The frozen v9 sources and compiled PDF are not overwritten.
