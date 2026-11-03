# Admitted Moving-Source Forecast Controls

## Material Passport And Question

This is a fresh real-Torch source-development experiment, not confirmation or
deployment. It uses only the 64 newly admitted publisher-train recordings.
The previous goal turn made progress: the complete DUT bridge and fit-only
admission finished. Those source checks are reused, not rerun. The immediate
question is whether a strong neural proposal transfers to this new source
after actual source-only fitting, before repeating any selected-risk critic.
An existing public architecture beating a control is supporting evidence,
not a new M3W architecture or the requested safe intervention contribution.

## Frozen Population And Inputs

Use all past-eligible targets, not just the 136,295 complete future windows.
The expected population is 195,286 complete-past agent rows, including partial
and wholly missing future labels, from the audited native 8/12 query ranges.
Keep the entire current-visible raw input inventory. Both comparator interfaces
consume the same nearest eight complete-past neighbors; partial-past current
agents are retained in a separate all-current observed-motion feature vector.
That 48-column vector is stored in memory for the later same-predictor cost
comparison, not fed to either forecaster or claimed as proven useful here.
It represents observed crowd/view motion, not calibrated camera ego-motion.

No future endpoint, target availability, track remaining length, central
velocity, test-endpoint goal, image-composite or audit future image enters the
model inputs. Future labels are used only in loss/evaluation. Exact future
perturbation/truncation tests cover the new encoder. Offline annotation-prefix
invariance does not prove sensor-time annotation provenance. Pixels and native
annotation steps are not metres/seconds; equal 8/12 counts do not align physical
durations to the European stride12 or SDD tasks. Native +50 is not run here.

Pack the complete matrices in RAM and retain only hashes/metadata on disk.
Source geometry, labels, positions and current inventory have separate paths.
Unknown labels are not zero errors. Retain every conservative source group;
never remove a difficult group to make endpoint FDE estimable.

## Complete Producer Design

Bind the admission's proposed three source folds without outcome-based changes.
For each of three actual seeds 17/29/43 and each fitting fold, train one PCT and
one fixed-head K=1 EqMotion: 18 primary fits. Each producer fits only that one
fold; both other folds are excluded. This supplies disjoint predictor/critic/
held-role combinations for the later critic experiment without teacher exposure.
Each target group has two excluded producers and three actual seeds per family.
Preprocessing, baseline selection and loss factors use producer fitting groups
only. No outside Moving clips, DUT or DroneCrowd predictive arrays are opened.

Architectures and the 4,000-update, batch64 AdamW budget match the existing
registered source controls. No pretrained weights or label-selected EqMotion
head. This is not published minADE20/minFDE20 reproduction or an isolated
equivariance/capacity test. The causal baseline family is unchanged: position,
backward-difference CV, .90/.97 damped velocity, history OLS4/OLS8. A producer
selects its baseline on fitting groups only. Both families have identical loss
factors, group sampling and draw coverage, checked at completion.

For each family's first job, run an actual 50+50 resume versus direct100 pilot;
require identical scientific checkpoint, optimizer/RNG/draws and outputs. The
resumed100 prefix can seed that same primary job without changing its identity
or total4,000updates. Record fresh versus reused updates. After all fits,
freshly refit the first full job of each family and require exact checkpoints.
Saved predictions must all reproduce in a separate full inference replay.

## Outcomes And Statistics

Registered primary: EqMotion-minus-PCT paired completion-lower difference in
percentage points using the fixed known CV denominator, equal source-group
aggregation after producer and actual seed averaging. Reuse the unchanged
triangle/disagreement completion calculation: wholly unknown rows have a
paired-error radius, not an imputed target. The known component uses each
row's available-label ADE; it is not full-future ground truth on partial rows.
Report known difference, per-model available ADE, endpoint FDE support, all
six motion controls and the fitting-selected strongest baseline. A test-best
causal baseline is diagnostic only and cannot become an inference choice.

Use all24conservative groups, 3,000 grouped paired resamples and actual seed
breakdowns. Groups are not certified IID physical sites; these are exposed
development crossfits, not independent confirmation. A positive forecast result
does not pass complete2%operational/easy risk or justify deployment. Selected
gain/harm, matched no-control/agent/uniform/joint policies and calibration are
not yet evaluated in this bank. All negative results remain reportable.

## Runtime And Completion

Use native arm64 .venv-pytorch, CPU4compute/1interop, no DataLoader workers or
Torch resource probing. Save atomic checkpoints and heartbeat every50updates.
A real-data pilot estimates speed, memory, serialized checkpoints and full
artifact footprint before starting the complete matrix. The preliminary model
state sizes are0.37MB PCT and12.47MB EqMotion; these are capacity checks, not
successful training. The local disk has about650MB above its original10GiB
reserve. RAM-only packing and compressed prediction files are required to
avoid a duplicate large cache. If actual resources fail, retain checkpoints
and use the authorized isolated CREATE path, not smaller science or fake
completion. Every invocation has a12hboundary and unchanged resume.

Freeze code/config before fitting. Completed receipts are immutable. Separate
preparation, pilot, fitting, inference and replay evidence; never call a profile
or cached result fresh training. Source admission and protected reservations,
the2%contract, Stage5C-off and SMC-off conditions remain unchanged. The full
research goal remains active/unmet.
