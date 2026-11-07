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

The latest eight-page PDF also records the stopped one-sided information-loss
diagnosis and the primary-literature scope recheck. Its replacement dual-view
model is still training: no new dual-view outcome is included. All eight current
rendered pages were visually inspected. Thirty-one focused checks pass; the
full historical suite was not rerun. The frozen v9 sources and PDF are unchanged.
