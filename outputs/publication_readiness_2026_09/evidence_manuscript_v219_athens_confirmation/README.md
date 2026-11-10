# Athens in the main comparison

The manuscript now includes the frozen four-date Athens evaluation.
It adds a positive comparison with CV without attributing that gain
to joint selection. The title, abstract, disclosure and existing
results are unchanged. The author's editable Word is not replaced.

The local candidate is in
`data/stage_cvpr2027_experiments/submission_admin/local_handoff_v219_athens/`:

- `paper.pdf`: eight pages, including references.
- `supplement.pdf`: eleven pages, with the Athens result in S10 and
  the cross-cohort comparison in S11. Existing plots retain their palette.
- `replay.zip`: offline scalar replay, with its inherited evidence chain.
- `manifest.json`: hashes binding these three files.

The source archive is in
`data/stage_cvpr2027_experiments/evidence_manuscript_v219_athens_confirmation/`.
Nothing has been uploaded or submitted.

## What the added result supports

The joint proposal's Athens sample envelope against CV is
[+0.02383, +0.08858] pp. Its lower endpoint has a descriptive
95% date-bootstrap interval of [+0.00622, +0.04144] pp.
Both joint and same-family proposals have positive lower bounds
on all four dates. Against independent selection, however, the
envelope is [-0.04427, +0.05304] pp. Equal per-step movement does
not resolve this comparison. Athens had no release calibration.

This evidence is **cached_verified**: the original freeze, roster
and readout hashes were checked, and the date-level aggregation was
replayed. The document builds and anonymous replay verification are
**fresh_run**. No new training or label access was performed.

## Verification

- Main PDF: eight pages; title, abstract and disclosure retained.
- Main text ends on page seven; new dataset citations render on page eight.
- Appendix: eleven pages; the content streams of the first nine pages
  are identical to the stored color-figure appendix.
- The missing loose v218 supplement was read from its hash-verified
  replay archive, not regenerated in the author's working directory.
- Standalone archive verification passed without site packages or network access.
- Twenty focused tests passed, including corrupted-role, source-pairing,
  interval and manuscript-protection checks.
- Main pages 1-8 and new appendix pages 10-11 were visually inspected.
- A read-only CREATE query confirmed that the four recorded M3W jobs
  37835856, 37835859, 37837636 and 38287008 ended COMPLETED, exit 0:0.
  No job was submitted, cancelled or restarted.

The pNEUMA article and release metadata were verified against the
[provider's terms](https://open-traffic.epfl.ch/index.php/terms-conditions/)
and [official release](https://zenodo.org/records/10491409). S10 includes
the requested data acknowledgment. The measured coordinate units are
unchanged; the provider's metadata is not a new calibration of our pipeline.

## Remaining priorities

1. **Selection evidence.** The main bottleneck remains causal ranking,
   not the absence of feasible replacements. A method improvement must
   beat matched independent and uniform controls on development data
   before it justifies opening another independent cohort.
2. **Release evidence.** Fixed-proposal utility does not show that a
   calibrated rule releases useful forecasts. WEPDTOF and Songdo joint
   release remain zero; Athens release is not run by protocol.
3. **Final author merge.** The edited Word must be reconciled with this
   separate evidence addition before the final PDF is checked and uploaded.

The defensible paper contribution is the controlled comparison and its
feasibility/selection distinction. The current results do not establish a
superior joint neural policy. Submission readiness remains false.

## Reproduce

```sh
python scripts/build_m3w_athens_candidate.py --compiler /path/to/tectonic
```

The builder uses the existing pypdf/reportlab document environment and the
cached official LaTeX style. It refuses to replace differing candidate files.
For the anonymous replay, extract `replay.zip` into an empty folder and run:

```sh
python3 -I -B -S verify.py
```

This recomputes aggregate checks, not model training or raw annotation joins.
[Main build](receipt.json) | [Package verification](review_receipt.json) |
[Chinese reproduction guide](../../../reproducibility/M3W_PAPER_REPLAY_ZH.md).
