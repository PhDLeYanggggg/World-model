# Frozen Severity Transport: Conclusions

## Evidence Status

Fresh error accounting and fixed-batch gradient calculation are complete for
144 frozen heads: six assignments, three seeds, full/motion-only inputs and
four source-held localities per combination. The models, source rows and
labels are cached_verified; this is not new training or an independent test.
There were zero optimizer updates, no policy changes and no new role access.
The parent severity-weighted auxiliary experiment remains a failed repair.

## Main Result

Excess error often concentrates within a held recording, but the registered
diagnostic does not establish widespread training-gradient domination or a
general radial feature-support explanation. These are different questions.

All counts below use known-label, positive-disagreement rows. Each comparison
has 72 dependent views; the views are not 72 independent scenes or replications.
The all-known-row accounting is also retained in every group artifact.

| Inputs / comparator | Views with worse easy-harm MSE | Of those, >=50% of positive row excess in one recording | Of those, radial-outside excess enrichment >20 percentage points |
|---|---:|---:|---:|
| Full / original cost model | 38/72 | 24/38 | 10/38 |
| Full / ordinary auxiliary | 45/72 | 26/45 | 8/45 |
| Motion only / original cost model | 44/72 | 30/44 | 2/44 |
| Motion only / ordinary auxiliary | 46/72 | 32/46 | 5/46 |

Positive and negative row excess are separated before recording/track sums.
The table is not the count of failed bootstrap intervals from the parent
experiment. The original comparator is the exact matched cost-only control.
An individual row can have positive excess even in a net-improving view.

## Fitting Concentration

At each final checkpoint, the saved 256-draw diagnostic fitting batch gives:

| Inputs | Median largest-recording norm share: ordinary BCE | Weighted BCE | Easy-harm cost | Weighted BCE >=50% |
|---|---:|---:|---:|---:|
| Full | 16.19% | 30.25% | 46.02% | 11/72 |
| Motion only | 17.42% | 38.83% | 56.41% | 26/72 |

The weighted loss is more concentrated descriptively, but only 11/72 full
views reach the registered threshold; more than half was required. The joint
recording-influence flag therefore stays false despite concentrated held
error. Training recordings and held recordings are disjoint: the figure
compares two concentration summaries, not influence by the same recording.

These gradients describe a fixed diagnostic batch at final parameters, not
the last optimizer minibatch or the whole training path. Group vectors add
to the total; their norms do not add to causal influence. Signed projections
and cancellation remain in the artifacts. See [scope note](batch_scope_note.md).

## Feature Support Proxy

For full/original views, median radial-outside row share is 3.50%, versus
4.83% of positive excess mass. Only 10/38 worsening views exceed the fixed
20-percentage-point enrichment rule. Motion-only medians are 4.71% of rows
and 0.15% of positive excess mass; only 2/44 worsening views are enriched.
The radial-association flag stays false. Medians describe separate
distributions and should not be divided to estimate a pooled enrichment.

The cut is the fitting-only, equal-locality-weighted 95th percentile of
unclipped standardized feature radius. It uses no held labels or statistics.
This cannot measure conditional support, exclude local domain shift or prove
missing information. Radius can be small while a particular causal pattern
is unsupported. No feature clipping was applied to the model.

## Decision

The diagnostic has not found sufficient evidence for a broad recording
resampling, gradient-surgery or radial-OOD repair. It does not prove these
mechanisms absent. Do not remove adverse recordings, weaken the original
comparator or tune thresholds on these source-held outcomes.

The next question is whether residual cost errors recur in identifiable
past-only motion/interaction contexts, including errors inside the radial
range. Use fitting-only context definitions and leave-locality-out source
checks to distinguish learnable conditional bias from unstable labels/support.
Any subsequent training repair must be registered separately; none has been
trained or shown effective here. Independent selection, risk calibration and
confirmation remain unopened. No model promotion is justified.

## Reproduction and Claims

The native-arm64 full calculation completed normally with per-group resume
and heartbeat. Its recorded 237.9788 seconds exclude registration/parent
preflight; the earlier reused pilot took 15.5903 group seconds, not end-to-end
command time. Replay/test details belong to `verification.json`, not to a
scientific pass gate. CREATE was checked read-only; no jobs were submitted
or changed. Raw data, caches, checkpoints and image previews stay private.

M3W remains a 2.5D trajectory/world-state research track, not true 3D or a
foundation model. Eight observed and twelve predicted annotation steps,
pixel/detector-derived labels: no metric, seconds, human-gold or physical-
safety claim. Stage5C and SMC remain off. Not yet submission-ready.
