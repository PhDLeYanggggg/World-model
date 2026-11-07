# Decision Manuscript v13

This five-page working draft adds the causal-only oracle, the
same-switch past-only comparison, and the new fixed-recipient
increment fit to v12. The abstract and new main-text section
report the negative result. The separate [v12 supplement](../evidence_manuscript_v12/supplement_readout.md)
still contains the unchanged risk definitions and source-influence
checks. No independent result is present.

Subsequent [matched-count missingness experiment](../native_visibility_increment_v1/README.md)
and [fixed-recipient risk proof](../native_fixed_recipient_risk_bound_v1/README.md)
are not yet incorporated in the frozen PDF.
The [exact-count feasible-versus-realized comparison](../native_exact_count_joint_capacity_v1/README.md)
adds a separate reproducible vector figure and is also not yet
incorporated in this version.

~~~sh
out=data/stage_cvpr2027_experiments/evidence_manuscript_v13
.venv-pytorch/bin/python scripts/build_m3w_decision_manuscript_v13.py \
  --source-zip "$out/manuscript_source.zip"
cd "$out/build"
tectonic -X compile manuscript.tex
~~~

The builder rechecks the three new result hashes against replay
receipts and verifies the matched-switch premise. The eight-file
local source ZIP excludes checkpoints, raw annotations, author
paths and the publisher template. Recompiling in an empty directory
using the verified official style produced identical five-page
rasters. Source ZIP SHA-256:
3bdaed9dcfe44e653a168deeeae4f2659bc2649318945790e222e90146ac3602.
This is typesetting reproduction, not end-to-end model reproduction.
See the [Chinese local and CREATE guide](../evidence_manuscript_v12/operation_zh.md)
for the stable runtime and approved independent protocol.
