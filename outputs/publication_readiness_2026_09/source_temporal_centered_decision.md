# Fixed Temporal-Centering Repair

## Material Passport

Source-only exploratory follow-up after the negative 36-head pretrained study,
commit37d337b0. Same approved offline 8-observed/12-predicted annotation-step
task and the same15,430stationary-history source queries. No main, bookstore,
sealed, calibration or confirmation forecast. No final-test tuning.

## Diagnosis And Hypothesis

Fresh past-only input audit aligns123,440history keys: none of the windows has
eight identical images, all have eight supported frames. Median person-box
footprint is9.34by11.86pixels in the32x32crop, and median box area is10.94%.
Image differences are present inside and outside boxes. Frozen feature energy
in within-window temporal variation averages2.5715%, median2.0464%.
The large shared component is shared appearance, not proven background or
causal explanation. Pixel change is not verified motion or intention.

Test whether removing shared appearance reduces source-scene fitting and
improves actual excluded-site trajectories. Two fixed input transforms:

1. centered: subtract the mean of the available eight past embeddings per query.
2. centered_unit: same, divided by per-query RMS L2 temporal variation, floor0.001.

Missing tokens remain zero; centering uses no future labels, fitted cohort
statistics or domain labels. Geometry and coverage are unchanged. RMS may
amplify nuisance changes; the second arm tests that possibility, not a guaranteed
repair. All transforms are observation-window operations, not future rollout.

## Fixed Design

Four sites(coupa/deathCircle/gates/hyang), seeds17/29/43, two arms =24new heads.
Each receives10,000updates, full-row uniform batch64, all-target normalized ADE,
including zero targets. Same frozen ResNet features and63,960-parameter GRU head
as prior arms, identical seeded initial state and sample draw streams. No encoder
fine-tuning. Same AdamW .0003,weight_decay.0001,first2000constant then cosine
to .01initial rate.240,000totalnewupdates. CPU4/inter-op1/workers0. Checkpoint
every200updates with exact resume. Named100updatepilot counts toward budget.
No outcome-dependent stopping, best-seed choice, hyperparameter or threshold search.

Original geometry/current/sequence controls are cached_verified, not retrained.
Primary actual metric remains equal-site ratio of mean normalized ADE vs stationary
CV; not a simple mean of site percentages. Report every arm/site/seed, hard slices,
absolute zero-target harm, tails, native-pixel ADE/FDE and each binary future oracle.
Seeds aggregate errors, not forecast ensembling. Same2,000physical-site bootstrap
draws,seed38113; conditional four explored sites/shared training,not confirmation.

Fixed contrasts: each new arm minus sequence; each minus geometry; centered_unit
minus centered. Report all five, not a selected favorable comparison. Reduced
damage alone is not a forecast contribution. Static easy percentage is undefined
against zero-error CV; no relabeling as a2%pass. No model deployment selected here.

## Interpretation And Resource Boundary

Existing main-fit96pixel detail and optical-flow experiments were negative; those
are not rerun or claimed to match this distinct source subset. This repair tests
a specific shared-appearance hypothesis rather than assuming resolution is enough.
Input audit is post-hoc descriptive; this design is frozen before any new fit.
Local64GiBfree observed, prior36heads took739summed fitting seconds, so local run
is reasonable. Previous CREATE publickey access failure remains unrepaired;
not a fresh scheduler observation, and no remote jobs submitted.

Dataset-local/annotation-pixel raw frames only, no metric/seconds/true3D/foundation
claims. Offline supplied histories may use later annotation controls; not strict
sensor-as-of. Raw data, images, features and checkpoints stay out ofGit.
Stage5C execution and SMC remain forbidden. Long-term research goal remains unmet.
