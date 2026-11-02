# Working Paper: Current Cost and Calibration Evidence

This revision brings the complete SOURCE cost-loss and unsuccessful independent
calibration results into one current narrative. It is a research draft, not a
submission-ready method. Earlier failed results remain in the frozen
[v8 paper and supplement](../evidence_manuscript_v8/README.md); those files and
their PDF receipt are unchanged. The current forest experiment is still
training in this revision and has no claimed readout.

## Evidence Included

- The nine-producer SOURCE-to-development controller comparison and its
  unsupported same-count joint effect.
- The limited absolute-cost gain, together with failed recording-risk checks.
- The complete18-fit harm-loss negative and fixed-bank versus selected-pool
  cost diagnoses, not just a favorable easy-net metric.
- Complete102-recording calibration: no common multiplier satisfies the
  original full risk contract; all47,736metric groups replay exactly.

The data roles, actual forecaster seeds, repeated recording views, partial
future labels and missing independent final confirmation are explicit. These
are cached_verified scientific results, not new experimental replications
created by writing this document. Prose, figure rendering and typesetting are
fresh work. The current paper has five main-body pages and one disclosure/
reference page in the unchanged official2027style. All six pages were inspected
as PNGs and the review is bound to the current PDF hash. Formatting does not
establish scientific readiness or complete anonymity.

## Files and Build

`manuscript.md` is the editable prose source. A standard Markdown AST produces
`manuscript.tex`; no regex-only Markdown parser is used. `build_sources.json`
binds scalar evidence, figure and official style identities.
`render_verification.json` records current-source compilation and visual review.
The PDF, page PNGs and third-party official style stay in the ignored private
`data/stage_cvpr2027_experiments/evidence_manuscript_v9/paper_build/` directory.
No identity, author profile, source data, cache or model weight is included.

Run from the project root. Use an environment with Matplotlib for the figure,
Node with `marked` for AST rendering, the installed LaTeX compiler helper, and
a Python environment containing `pypdf` for verification. The locked Torch
environment does not include `pypdf`; this run used the configured bundled
document Python for that step rather than altering the training environment.

```sh
.venv-pytorch/bin/python scripts/render_m3w_working_paper_v9.py
NODE_PATH=/path/to/node_modules CVPR_TEMPLATE_DIR=/private/official/template \
  node scripts/build_m3w_working_draft_v9.mjs
```

Compile the private `paper_build/main.tex` with the installed `compile_latex.py`
helper and `--compiler tectonic --json`; preserve its JSON output in a uniquely
named log. Render every current page with `pdftoppm -png`, then run:

```sh
python scripts/verify_m3w_working_draft_v9.py \
  --compile-log /private/current_compile.log \
  --visual-reviewed-sha SHA256_OF_THE_PDF_ACTUALLY_INSPECTED
```

Do not supply the visual flag without inspecting every page of that exact PDF.
Regeneration clears the compile flag and requires current rendering/review.
The upstream `lineno.sty` UTF-8 warning is retained and disclosed; no final-pass
overfull box, unresolved reference, missing glyph or font-shape fallback occurs.
The first attempt at source generation used the wrong calibration-receipt
filename and did not produce TeX; its failed compile log is retained privately.
The corrected builder binds the actual `verify.json`, not a fabricated receipt.

## Still Missing

A learned full-policy risk pass, supported scene-joint gain, matched public
forecasting implementations, independent final confirmation and a complete
anonymous cold-start reproduction remain incomplete. No new JEPA or Transformer
dynamics contribution is claimed by this revision. Available-label ADE is not
full12 ADE; bounds omit missing steps inside partial futures. Local pixels and
raw-frame steps are not metric or seconds. Silver is not human gold. Stage5C
and SMC remain off. No registration or paper submission was made.
