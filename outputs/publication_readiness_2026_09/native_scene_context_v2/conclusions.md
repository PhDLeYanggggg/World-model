# Identity-Resolved Scene Context for Joint Intervention

## Material Passport

2026-09-21. `fresh_run`: raw-annotation identity reconstruction and complete
visible past-context cache over the admitted four source scenes. `cached_verified`:
frozen 8-to-12 cohort, past geometry, query identities and source hashes.
`not_run`: a new neural model, joint policy comparison, independent calibration,
new deployment and untouched confirmation. This is an input repair enabling the
next method experiment, not a prediction improvement or publication gate.

## Completed Repair

The private cache contains 175,756 eligible pedestrian forecast targets across
33 recordings and 20,932 recording/frame groups. Every target has its reversible
past-only transform. Every visible source agent at those frames is retained:
321,561 agent/frame context rows, including 145,805 without a neural prediction.
Those context rows are not unique people or independent situations.

Of the context-only rows, 127,254 are non-pedestrian and 18,551 are pedestrians
without eight consecutive sampled past observations. This exactly accounts for
the difference from the target population, without looking at future support.
Among all context rows, 5,748 lack two observations needed for causal velocity.
Their background CV forecast mask is false. Stored placeholder coordinates must
not be interpreted as zero velocity, forecast evidence or physical safety.

The previous positional assembly probe contained 52 false unique identity
links and 3,036 ambiguous neighbor slots. The repair resolves all observed
neighbor identities from `(recording, current_frame, source_agent_id)` and the
same causal distance/agent-ID ordering used by the original input builder.
Of 1,325,792 observed neighbor slots, 869,241 point to an eligible target query;
456,551 point to a context-only agent. No context-only identity is substituted
with a coincident forecast target. The repair does not change the frozen neural
predictions or their existing reported metrics.

Full visible *past context* is now available, not full-scene neural prediction.
Irregular histories remain explicit: 7,880 neighbor-slot histories contain
nonconsecutive observations. They are not silently resampled or extrapolated
backward to fill missing history. CV uses the actual two past timestamps where
available; all outputs retain the raw annotation-frame interpretation.

## Verification

All 10,175,375 observed neighbor history points reproduce the frozen input
geometry, masks and timestamps within float32 representation precision.
297 fixed scalar nearest-neighbor ID checks and 99 raw-source future-truncation
checks pass. Current source rows and histories do not change when later raw
annotations are removed. Target eligibility exactly matches pedestrian type
and eight consecutive past samples, not future-label completeness.

A separate verifier checks every context key, target link, neighbor link,
history mask/time and supported CV forecast. Its scalar CV arithmetic differs
by at most 4.55e-13 pixels from the vectorized implementation. Verification is
an independent implementation by the same assistant, not an external reviewer
or independent research test. 139 related tests pass; the full historical
repository suite was not rerun. Cache replay preserves the analysis hash.

The first context implementation inherited unused target arrays through a
general parent loader. It did not use them to build context, but its metadata
incorrectly claimed no target read. Version2 replaces that loader with an explicit
geometry/query-key allowlist and reruns all33records. No future target array is
loaded in construction or separate verification. The v1 attempt remains local
and is not the authoritative evidence artifact. The alignment/support audit is
different: it intentionally reads previously available targets for diagnostics,
never for selection. This distinction is preserved in the receipts.

## What This Changes Scientifically

The simple safety control still lacks independent calibration and retains
unknown selected outcomes. The new cache repairs the prerequisite for testing
scene coupling: shared coordinates, correct identities, all current context and
an explicit unsupported-prediction mask. It does not establish that interaction
helps, that background CV is accurate, or that a proximity proxy is a collision
or physical-safety label.

The next meaningful experiment is a fixed source-only comparison of no control,
independent selection, whole-scene selection, geometry-aware independent control
and actual joint coupling with the same forecasts and allocation budget. Keep
unsupported background predictions visible and report their coverage. Separate
better unary geometry from true pairwise interaction; do not claim coupling from
a reduced switch rate or a newly tuned threshold. Independent calibration and
confirmation require clean, separate source roles before any safety claim.

All four sources remain design-exposed. Native pixels and sampled annotation
steps only; no seconds, metric, true3D, foundation or human-gold assertion.
Stage5C/SMC remain off. No new deployable model is promoted.
