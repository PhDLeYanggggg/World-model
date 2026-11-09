# A reading guide for the supplement

I kept the original supplement and added two opening pages to a separate
local copy. The first page links directly to the forecast controls, risk
definition, OpenDD and MovingDroneCrowd++ diagnostics, WEPDTOF comparison,
and source roles. The second page reports the
[direct pair-effect comparison](../opendd_direct_pair_effect_v1/README.md)
at identical forecast menus, counts, and peak displacements.

The paper and original supplement are unchanged. All 50 original pages
rendered identically at 72 DPI after adding the two pages. The new file keeps
the original outline, all 275 named destinations, and the original citation
link. Nine new links were checked against their PDF page references. Both
new pages were inspected as rendered images. A targeted scan found no
author name, institutional domain, local home path or project owner handle
in the extracted text, metadata or active links of either submission PDF.
This is a packaging check, not an independent scientific result.

## Rebuild the local option

Use Python with `pypdf` and `reportlab` installed. The script refuses to
overwrite either PDF.

```sh
base=data/stage_cvpr2027_experiments
python scripts/build_m3w_supplement_guide.py \
  "$base/evidence_manuscript_v165/supplement/supplement.pdf" \
  outputs/publication_readiness_2026_09/opendd_direct_pair_effect_v1/readout.json \
  "$base/submission_admin/supplement_with_evidence_guide.pdf"
```

The byte-identical rebuild has SHA-256
`b36fede903ed779fb49ac23138bb5940331fcba3ecffa24a44ef2dc621ddae40`.
The PDF is local only. No upload, author record or policy was changed.
