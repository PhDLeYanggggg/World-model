# Native Cost-Learning Data Ready

## Material Passport

- Date: 2026-09-21; scope: exploratory source-only producer-lineage repair.
- fresh_run: 18 neural fits, prediction inference, cost reduction and partition export.
- cached_verified: source inputs and 12 existing outer-held models/predictions.
- not_run: cost-head training, risk calibration, threshold selection and independent confirmation.
- Status: prerequisite complete; no new accuracy, safety or deployment claim.

## What Completed

All six pairs of excluded source sites and seeds 17/29/43 completed their fixed
4,000-update budgets. These are 18 real arm64 Torch fits, 72,000 optimizer
updates and 4,608,000 row draws. Summed fitting time was 1,525.214 seconds
(25.42 minutes). The 100-update pilot was resumed and counted once, not discarded.
Each model has 88,514 parameters and learns from the other two physical sites.
Training sampling visited 80.09%-99.34% of its eligible rows; indexing the full
population does not mean every row was sampled. Held-site training draws are zero.

The twelve existing single-held native predictors remain hash-bound and are
reused without refitting. New pair-excluded predictors satisfy 36 ordered
outer/inner requirements: the future head's validation site and the training
row's own site are both absent from its upstream predictor and fitted normalizers.
Pair reuse in two directions is not counted as two fits.

## Cost-Training Views

The experiment retains 175,756 unique source queries from four already explored
sites. Eighteen prediction caches generate 1,581,804 producer/seed entries,
exactly nine entries per source query across the full matrix. These are not
1.58 million independent samples. Of these entries, 1,556,613 have supported
masked ADE; 25,191 remain unknown. Unknown cost is never replaced with zero.

Each outer site has three seed-specific training views:

| Excluded Outer Site | Rows per View | Supported Costs | Unknown Costs | Outer Rows Included |
| --- | ---: | ---: | ---: | ---: |
| coupa | 147,696 | 145,179 | 2,517 | 0 |
| deathCircle | 139,171 | 137,197 | 1,974 | 0 |
| gates | 155,034 | 152,721 | 2,313 | 0 |
| hyang | 85,367 | 83,774 | 1,593 | 0 |

The exporter physically separates `inputs.npz` from `targets.npz`. The input
archive contains only row alignment IDs and causal predictions; IDs are not
model features. Past history/neighbor geometry is referenced through the bound
global row index and past-only packer. ADE/FDE, gain, benefit, harm and future
support metadata remain supervision only. Whole pair caches must not be fed to
a head because they contain both held directions; use the explicit view.

## Verification

272 data/code/parent bindings were checked. Independent arithmetic recomputed all
1,581,804 paired cost entries. Fixed first/middle/last inference batches from all
18 final checkpoints replayed 5,484 predictions exactly. Static-history outputs
matched the causal baseline exactly. All 36 upstream exclusion checks passed;
the twelve physical exports contain zero outer rows and replay exactly.
64 scoped tests passed, including exact resume, two-held-site poisoning,
indirect exposure rejection, unknown support and outer-payload poisoning.
The broad historical report-writing test suite was not rerun.

Machine-readable evidence: [analysis](analysis.json),
[checkpoint/scalar verification](verification_with_replay.json),
[physical exports](materialized_views.json). Analysis SHA-256:
`e9215760af5fe10b597802c5ada5c86071ce075be0973a413c8e95d815de3a2d`.

## What This Does Not Establish

All four sites have influenced research design. Clean fitting exclusions cannot
restore independent confirmation. The two-site training producers also differ
in training support from the three-site outer predictors; downstream cost
reliability must be measured rather than assumed. Partial-future ADE does not
identify errors on unobserved future positions. No selector, conditional-risk
model, calibration threshold, multimodal contribution or new deployment has
been validated by these caches.

The chosen route is native forecasting plus learned benefit/harm and conservative
intervention, not scope expansion into a claimed foundation model. The author
delegated the choice; [strict zero-reference protection](research_choice.md) is
retained prospectively without a convenient new pixel tolerance. Frozen earlier
pending statements remain historical. The next experiment must freeze the
native easy-group definition and matched cost-head comparison before fitting;
independent calibration remains separate. No repeated question about the already
resolved primary metric or strict-vs-relaxed choice is needed.

SDD pixel/raw-frame only. No metric, seconds-level, true-3D or foundation claim.
Stage5C and SMC remain disabled. The long-term research goal remains unmet.
