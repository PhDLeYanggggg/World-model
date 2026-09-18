# Current-Cohort Event Support Audit

This is a post-hoc diagnostic, not a new training comparison or confirmation test.
It follows the negative temporal-centering experiment, commit 4fdeb9e2. The old
full-SDD support census is retained, not rerun or claimed as new evidence.

## Fixed Scope Before Calculation

Use all 15,430 current stationary-history source queries in coupa, deathCircle,
gates and hyang. Bookstore/main/outer roles remain excluded. The approved offline
eight-observation/twelve-prediction annotation task at stride 12 is unchanged.
No filtering, threshold tuning, model fitting or new forecast is performed.

Link each query to its maximal preceding contiguous, non-lost run with identical
annotation center. Recording, local track and run-start frame define an episode
group. Raw gaps, lost observations or a center change break a run. This group key
uses past rows only. It is not a verified physical event or independent person.

Separately describe labels: any sampled future change, excursion above 10 pixels,
excursion at least half the past-box diagonal, and persistence outside that radius
in the last four predicted steps. These reuse previous audit definitions, not
newly selected targets. Inspect raw-frame changes within the next 144 annotation
frames for grid-aliasing/missingness; future fields never enter past features.

Report windows, tracks, recordings, groups and disjoint full-window spans.
Show contribution concentration and current/complete/moving-neighbor support,
past image coverage/feature variation and source-control support. Cluster
concentration is descriptive, not an inferential effective sample size. Four
shared scenes do not become many independent scenes by counting tracks or groups.

As a diagnostic only, reweight the five already frozen trajectory arms by
past-defined episode and by scoped track, then aggregate equally across sites.
Retain the original window-based equal-site score. No new predictions, checkpoint
choice or altered main metric; a favorable retrospective weighting cannot become
a deployment claim. Report all arms and all three fixed seeds.

Future mutation and truncation must preserve past-episode features. Hash-check
prior aligned rows and image audit. Recompute the new audit to verify its outputs.
Keep row-level traces private; publish aggregate counts and reproducible code.

The result may motivate independent support acquisition or an input repair, but
does not authorize removing hard/static rows, changing the primary metric,
relabeling explored data as confirmation, or training another threshold sweep.
Dataset-local/raw-frame only, offline annotations not strict sensor-as-of,
labels not human gold, Stage5C and SMC off. No CREATE job is required for this
small local audit; previous SSH authentication failure is not a live job state.
