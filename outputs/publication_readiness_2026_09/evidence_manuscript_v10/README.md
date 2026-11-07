# Native Moving-Drone Working Manuscript

This version separates the native MovingDroneCrowd++ observation8/prediction12
task at stride1 from the earlier European stride12 study and historical SDD
raw-frame studies. Their absolute errors and gains are not pooled.

The draft integrates completed support, prefix-loss and past-image cost-head
experiments, including their negative registered contrasts and failed original
recording-risk checks. All twelve past-image actors complete4000updates and
both fixed-design full refits reproduce exactly. Their primary whole12known
effect is-1.804pp and completion lower-10.931pp[-17.826,-5.288];the component
screen fails. These outcomes and frozen-checkpoint diagnostics are included,
with their source hashes. No new outcome scoring is done by the manuscript builder.

The completed query-view experiment adds six full4,000update actors,a full
identity control and one complete first-design treatment refit. All twelve
actor/camera-CV forecast banks and scalars replay exactly. Its primary known
effect is-5.133pp,paired lower-21.447pp[-33.682,-9.911],with3/14positive
known sources. The camera-CV secondary control does not replace this failed
primary. All six final matched training minibatches are worse;this is not a
training-average loss estimate or identification of a unique causal failure.
The draft preserves the failure and original risk/support/NA rules.

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
committed. A six-file portable aggregate package now replays the three completed
native actor primary analyses in a clean stdlib-only process. It preserves the
original bootstrap indices and negative outcomes; it is not complete training
or prediction reproduction, independent confirmation or guaranteed anonymity.
Full model-training reconstruction is still outstanding.

The eight-page PDF records the stopped one-sided information-loss diagnosis,
the literature recheck and the completed replacement dual-view experiment.
Both registered dual-view screens fail despite exact twelve-bank/scalar replay.
The whole-12 known contrasts are +0.828pp versus the matched native branch and
-0.321pp versus the original actor; the completion lower intervals are entirely
negative. Pages 1 through 5 are identical to the inspected previous render;
changed pages 6 through 8 and the new source/training figure were inspected.
Forty-three focused checks pass; the full historical suite was not rerun.
The frozen v9 sources and PDF are unchanged.

The 7 November revision retains eight pages and includes the negative
matched-count pair comparison plus a separate three-seed predictor control.
The full 24-group predictor primary remains not estimable. The 23 supported
groups are a secondary result, not evidence of an independently confirmed
three-seed intervention policy. The named LLM Usage Statement now precedes
References. Pages 6 through 8 were inspected; pages 1 through 5 were identical
to the last inspected render. Thirty-five focused checks passed for this update.

The later seed-disagreement revision adds the source-excluded matched-count
cost experiment. The conservative utility contrast is -3.535 pp with an
entirely negative exploratory source interval. The PDF still has eight pages;
the main text ends on page seven, followed by the disclosure and references.
Changed pages were inspected and 25 focused checks pass. This is still a
working draft, not a deployment or submission claim.

The recording-risk revision adds a direct comparison with OLS4 and the
unchanged selected-reference gate on all 34 exposed recordings and three
forecast seeds. The learned anchor passes no recording view. The eight-page
PDF is checked again: pages 1 through 6 are pixel-identical to the prior render,
and pages 7 and 8 were inspected. The mandatory disclosure still precedes
References. Twenty-seven focused tests pass; no new independent result.

## Evaluation Supplement

`supplementary_math.md` and its generated TeX give the exact scope of the
registered row-support completion envelope. Partially labelled rows retain
their observed-step loss; only wholly unknown row contrasts are bounded, with
the known CV reference denominator fixed. This is not a full-grid ADE bound.
A separately documented existing full-grid calculation bounds every missing
step. Elementary proofs, attainability examples and synthetic tests keep these
quantities distinct without changing any registered metric or historical score.

The synthetic checks and manuscript checks form part of the 43-test verified
suite. The three-page supplementary PDF compiles and every rendered page was
inspected. All 15 active registered model/protocol dependency hashes remain
unchanged. The dual-view disk interruption recovered for exactly the remaining
65 updates in the tenth fit, with final RNG/draw parity; that engineering
success is separate from the failed predictive comparisons.

```sh
.venv-pytorch/bin/python scripts/build_m3w_native_supplement.py
python3 /Users/yangyue/.codex/plugins/cache/openai-bundled/latex/0.2.8/scripts/compile_latex.py \
  data/stage_cvpr2027_experiments/evidence_manuscript_v10/supplement_build/main.tex \
  --compiler tectonic --json
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_bound_estimands.py tests/test_m3w_native_manuscript.py
```
