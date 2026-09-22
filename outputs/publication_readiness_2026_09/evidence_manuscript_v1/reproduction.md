# Manuscript Reconstruction and Evidence Scope

## What Ran

Fresh work in this revision: structured English manuscript, aggregate-table
arithmetic, five fixed paired contrasts, all twelve easy scene/seed values, and
a scientific vector figure. The six source reports are byte-hash verified against
the recorded completed versions. Raw tracks, predictions, new labels, model
weights, and pending external annotations are not read by the builder.

This is `fresh_paper_export_of_cached_verified_aggregate_reports`. It is not
fresh model training, fresh bootstrap, independent replication, risk calibration,
data admission, or a new performance result. Historical checkpoint replay and
separate arithmetic are described in the original experiment records, not rerun
or relabeled by this packaging step.

```bash
.venv-pytorch/bin/python scripts/build_m3w_evidence_manuscript.py
.venv-pytorch/bin/python scripts/build_m3w_evidence_manuscript.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_evidence_manuscript.py
```

For visual inspection, the builder also accepts `--preview /tmp/m3w-paper.png`.
The PNG preview is local only; the tracked SVG is an aggregate scientific plot,
not a third-party image or raw data. `--verify` compares the evidence JSON, CSV,
and Markdown tables, not raster pixels or all manuscript prose. The figure is
checked visually. The focused tests do not replace the nonhermetic legacy suite.

## Inputs and Traceability

`evidence.json` contains six pinned analysis hashes, per-row JSON pointers, both
positive and negative contrasts, scene/seed damage, and no-training/no-deployment
flags. `main_table.csv` and the32-row `site_metrics.csv` are light aggregate data.
The latter retains raw pixel ADE/FDE, p95/p99 and outcome-support counts by site.
`tables.md` includes conditional
intervals and worst-site error. Historical experiment source locators:

- [Native predictor experiment](../native_forecast_v1/conclusions.md).
- [Adapted EqMotion comparison](../native_eqmotion_v1/conclusions.md).
- [Negative joint controls](../native_joint_controls_v1/conclusions.md).
- [Equal-budget objective comparison](../cost_budget_matched_v1/conclusions.md).
- [Latest region-weighted repair](../conditional_cost_v1/conclusions.md).
- [Calibration ancestor-exposure audit](../calibration_support_v1/conclusions.md).

The first builder attempt revealed the exact paired-contrast keys differed from
the assumed names. Both affected export tests failed before any figure was made;
the pointers were corrected to the existing schema. No source result was changed.
The first five focused tests then passed; a sixth checks the manuscript's rounded
table directly against the artifact-derived values. The final six-test run
passes, four text exports replay exactly, and the figure is visually checked.
A local
Matplotlib font-cache warning caused initialization work, not a training hang.

## Claim Review

The old chronological draft mixed multiple populations, metrics, and successive
abstracts. This manuscript selects one coherent completed development sequence
without selecting a favorable seed or redefining a failed gate. Earlier evidence
is preserved in the old draft and source reports; it is not reclassified as an
independent test. The latest abstract now agrees with the latest protection failure.

The native-loss-versus-old-loss 5.558% contrast is a **relative reduction against
the old model**, not 7.633 minus 2.183 percentage points. Equal-count and joint
controls remain explicit. Fraction-cost's lower observed damage does not become
a post-hoc deployment recommendation. Aggregate protection never overrides the
every-scene/seed criterion. The CI unit remains four design-exposed sites.

Primary-source verification on 22 September: EqMotion and JFP author metadata/
abstracts reopened; Mao et al. proceedings metadata and arXiv v1 section 4;
Shah et al. proceedings metadata and sections 2.2/3.1 from the original PDF;
Learn then Test v5 section 1.1. Previous scoped JFP mechanism review is retained
in the linked research record. The CVF EqMotion page returned403, so the author
arXiv record supplies the checked metadata. No full-paper human-read attestation
or exhaustive novelty search is claimed. The SDD author release page and author
PDF search metadata supply its dataset citation; the11.9MB PDF could not be
opened by the web reader, so a full-paper read is not claimed. URLs appear in the manuscript.

## Resource and Next-Action Record

The local committed source before this revision is e9c18a41, with existing
artifacts available. This revision needs no Torch training or new HPC job.
Local free space remains about54GiB at the check. Current SSH configuration has
no CREATE alias; no verified M3W remote path was recovered from the inspected
project inventory/configuration. No new remote login, queue inspection, or claim
of absent remote assets is made. The historical authentication failure remains
historical. The tutorial retains the recovery and scheduler-inspection procedure.

Highest-priority blockers are independent data roles/producer exclusion, failed
per-scene protection, and an unproven intervention contribution. The two earlier
questions remain pending: independent-scene acquisition versus strict nested
development, and permission at the annotation-archive download warning. This
revision does not repeat them or bypass either. Stage5C and SMC remain off.

Next: complete permitted source intake, agree data roles, then prospectively test
conditional harm on truly out-of-fitting selection regions. Do not keep polishing
this draft instead of obtaining the missing scientific evidence. Submission is
not ready and the long-term goal remains active.
