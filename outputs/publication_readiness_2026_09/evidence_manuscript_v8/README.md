# Working Manuscript: Partial-Label Forecast Substitution

This draft asks when a learned forecast should replace a strong motion baseline.
It separates average benefit, positive harm and missing-outcome uncertainty,
and keeps unsuccessful controllers in the results. The current evidence is
exposed SOURCE development, not independent confirmation or a deployable method.
The broader multimodal agent-scene world-model goal remains unproven.

The source includes the earlier frozen-bank comparisons and the complete
zero-update coherent-cost control. The own-decision training comparison is now
complete and numerically verified; its observed gain and failed primary/risk
requirements are included together. The completed, capacity-matched ADE versus
published-deferral comparison is also included with both known and completion-
lower effects. The working supplement retains the two-sided guard failure
diagnosis. The complete two-sided boundary-loss continuation is now included:
observed routing improves, but required recording risk worsens and the primary
lower interval crosses zero. Its event-versus-cost limitation is retained in
the supplement. A successful build does not improve the scientific status.

## Files

- `manuscript.md`: readable draft generated from frozen v7 prose and verified results.
- `manuscript.tex`: anonymous two-column source using the official 2027 style.
- `build_sources.json`: source/result/style identities and compile status.
- `render_verification.json`: exact-PDF layout and recorded visual-review checks.
- `supplement.md`: current target/loss/support technical notes; a working Markdown
  appendix, not included in the checked main-paper PDF page count.

The locally built PDF, page renders, official template and third-party packages
remain private and excluded from Git. No author profile, email, raw data,
trajectory cache, checkpoint or third-party style file is included here.
The mandatory LLM usage statement is retained; owner-written project wording
does not replace truthful assistance disclosure.

## Build

Run from the repository root. This is a report build from public scalar evidence,
not training, new inference or an independent experimental replication.

1. Obtain the official [CVPR 2027 author kit](https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip)
   and extract it privately. Set `CVPR_TEMPLATE_DIR` to the directory containing
   `cvpr.sty`. The checked style identity is recorded in `build_sources.json`.
2. Use Node with `marked@17.0.5` available through `NODE_PATH`. The checked local
   environment uses Tectonic 0.17.0, pypdf 6.10.0 and matplotlib 3.10.9.
   They are build dependencies, not new scientific models.
3. Generate sources and paper-scale figures:

```sh
node scripts/build_m3w_typeset_draft.mjs
.venv-pytorch/bin/python scripts/render_m3w_cost_manuscript_figure.py \
  --output data/stage_cvpr2027_experiments/european_own_decision_cost_v1/paper_build
```

4. Compile `paper_build/main.tex` with Tectonic and retain the complete JSON
   compile log as `paper_compile.log`. The local build uses the installed
   `latex-compile` helper's `--compiler tectonic --json` path. Rebuilding the
   Markdown/TeX resets `compiled` to false; the verification step binds it again
   only after a current-source successful compile.
5. Render every current PDF page with `pdftoppm -png`, naming the output prefix
   `paper_build/page`. Inspect every page, including figures and references.
6. Run `scripts/verify_m3w_typeset_draft.py` in an environment containing pypdf.
   Only after inspecting all pages, provide `--visual-reviewed-sha` with that
   exact PDF's SHA256. A stale screenshot or PDF identity is rejected.

The current local checked build has seven main-body pages and one additional page for
the LLM statement/references. Overfull boxes, font-shape fallbacks, missing
glyphs, duplicate reference headings and unresolved final-pass references are
checked. An upstream `lineno.sty` UTF-8 warning is disclosed, not silently removed.
These checks are not a complete anonymity, venue-policy or scientific review.
Whenever results or source text change, regenerate, compile and inspect again.

## Research Status

The controlled cost study does not establish new JEPA/Transformer dynamics,
independent forecasting accuracy, cross-dataset safety, metric geometry or
seconds-level prediction. EuropeanSquares remains detector-silver/local-image,
obs8/pred12/rawstride12 in this pipeline. Three cost seeds share forecaster43.
Stage5C and SMC remain off. Actual submission requires the author's confirmation.
