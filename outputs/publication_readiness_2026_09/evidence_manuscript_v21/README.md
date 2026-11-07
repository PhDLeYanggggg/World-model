# Decision Supplement v21

[v22](../evidence_manuscript_v22/README.md) adds the constructive
completion check to the main draft and supplement.

The companion to the unchanged [v20 main manuscript](../evidence_manuscript_v20/README.md)
adds a short [shared-target sensitivity](../native_shared_pair_completion_v1/README.md).
The new bound narrows the completion envelope without establishing a
conservative joint advantage. The main metric and original external
admission rule are unchanged.

```sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v21
.venv-pytorch/bin/python scripts/build_m3w_decision_supplement_v21.py \
  --source-zip "$out/supplement_source.zip"
cd "$out/supplement"
tectonic -X compile supplement.tex
```

The builder verifies the shared-target aggregate against its replay
receipt and retains the previous scalar and template bindings.
The anonymous archive contains only TeX and a short build note.
With the verified style restored, the clean-source build matches
all five page rasters. Source ZIP SHA-256:
`7b530750b4ebaa61a61c02378fa3c5953d016750d57f5b1b4c770df0808f7ffd`.
The supplement still reports exposed development only.

The separate [observed-neighbor follow-up](../native_anchored_pair_completion_v1/README.md)
further narrows the same bound without changing the interpretation.
It is not part of the frozen v21 PDF.
