# Native Scene Alignment and Frozen-Control Label Support

## Material Passport

2026-09-21. `fresh_run`: common-coordinate reconstruction and frozen-control
support reductions on all 175,756 admitted source queries. `cached_verified`:
frozen forecasters, gain scores, geometry and past annotation-box provenance.
`not_run`: new model fitting, joint selection, independent calibration and new
confirmation. The four physical source scenes remain design-exposed. No closed
original validation/test, main, external or bookstore role was opened.

## Common Coordinates

All 33 recordings can be reconstructed from eight past annotation boxes, scoped
agent/frame keys and the original causal heading/path scale. There are 20,932
recording/frame query groups. Maximum stored-scale past reconstruction error is
0.00004662 annotation pixels. Restoring predictions and targets to the common
frame changes native ADE arithmetic by at most 9.095e-13. The original error
definition and strict exact-zero reference criterion are unchanged.

The original source computes a float64 coordinate scale, then stores the causal
feature scale as float32. Both are retained: annotation scale for source-context
reconstruction and the historical stored scale for exact metric continuity.
Small numeric QA tolerances do not authorize pixel-error safety allowances.

## Positional Linking Failed

The new scene-assembly probe tried matching neighbor positions to eligible
target queries. Unique current-position matches can still be false identities:
a non-target agent can coincide with a target, which is invisible to a target-only
lookup. The largest linked-history discrepancy is 11.6726 pixels. Another 3,036
slots are ambiguous within the target population. This index is not approved for
neighbor identity inference; its `target_neighbor_links_identity_verified` is
false. This is a failure of the new assembly probe, not evidence that previous
trained models consumed these proposed links.

The raw-ID repair is completed separately in
[native_scene_context_v2](../native_scene_context_v2/conclusions.md). That cache,
not the positional links here, must be used for subsequent scene experiments.

## Missing Outcomes in the Simple Control

The frozen past-stop/MSE-strict choices reproduce every prior choice hash.
Across three seeds they select 14,576 query/seed instances: 11,512 complete
twelve-step outcomes, 2,781 partially observed outcomes and 283 with no ADE label.
These are not independent events. The union is 7,794 unique indexed queries on
1,365 recording-scoped tracks and 4,434 scene queries; it includes 1,491 partial
and 154 absent outcomes. Overlapping windows remain dependent.

On the selected *supported* rows, mean native ADE harm is negative in each of
the twelve source/seed views. Complete-outcome means are also negative, but
neither proves the unobserved outcomes are benign. Future completeness is not
an inference feature, and missing-at-random transport has not been established.
The existing observed zero-reference protection is therefore not a population
safety guarantee or independent calibration result.

There are 1,074, 663 and 1,368 scene queries with at least two selected targets
for seeds17/29/43, respectively. This provides actual joint-decision opportunities,
not proof that coupling improves a decision. Most intervention groups do not
include forecasts for all visible agents: 3,319/3,616, 2,198/2,324 and 2,790/2,826
groups with an intervention have incomplete target coverage.

## Consequence

Do not build a scene-level claim from independently centered trajectories or
silently omit context-only agents. First retain all visible current context,
explicit source identities and causal forecast-support masks. Then freeze a
matched joint-versus-geometry-aware-independent experiment. Do not remove rows
using future completeness, retune the demonstrated selector, or count the
alignment repair as improved neural dynamics.

Eight observed/twelve predicted sampled annotation steps; native SDD pixels.
Raw-frame t50 remains supplementary. No metric/seconds/true3D/foundation claim;
offline silver annotations are not verified real-time perception or human gold.
Stage5C and SMC remain off.
