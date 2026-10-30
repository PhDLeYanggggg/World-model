# Selected-action query risk critic: fixed paired experiment

## Material Passport

Previous goal turn: progress. The complete reoptimized-bank comparison and its
separate replay establish modest fitting-time neural value, not neural-specific
joint synergy or safe deployment. This experiment addresses the remaining
selected-population harm error with fresh real Torch training. Code, config,
tests and this protocol are committed before any pilot or fit. Independent
roles remain closed. This is SOURCE fitting/resubstitution, not validation-
selected deployment or whole-pipeline out-of-fold confirmation.

## Distinct question and controls

Can a risk head conditioned on the original joint policy's selected candidate
set predict harm more usefully than a parameter-matched row-local risk head?
Use the original seven-action joint-matched policy, not the smaller shared-
physical-count control. Keep every original source row and candidate. A veto-
only policy cannot repair recordings where the original policy never switches,
so both models score all seven causal candidates and use the same allocator.
They may reassign actions; no label or outcome mask controls eligibility.

Two arms each have30,726 trainable parameters. Each encodes the same462 frozen
causal candidate/history/scene fields, log1p original predicted five costs,
candidate ID and log1p causal disagreement envelope. Both receive the same
initial selected-count metadata. The row-local arm pairs its candidate hidden
vector with itself. The selected-set arm pairs that vector with the mean hidden
vector of the original policy's selected actions in the same current query.
An empty selected set has a zero context and an explicit zero count; it is not
discarded. No future-known mask enters the pool. Record/frame/agent identities
only group and align inputs; they are not learned ID embeddings.

This is not the old all-peer mean experiment: it uses actual selected action
embeddings and all seven candidate costs, with a new common selected-query loss.
It is not old coarse-cell calibration, a new global lambda sweep or a trajectory
residual. Both arms change the same architecture/objective relative to the
parent, so a parent improvement alone cannot identify selected-context value.
The paired row-local comparison isolates the pooling difference.

## Protected values, targets and objective

The actual forecast bank, original cost heads and original benefit/reference/
easy-reference predictions stay frozen. A zero-output, anchor-exact bounded
rational link adjusts only positive harm and positive easy harm. Its envelope
and easy<=all-harm relationships remain. Original cost predictions and context
are inputs, not oracle best-action labels.

Train on all causal current queries, including unknown-outcome rows. Known
available action costs supervise the two risk channels; unknown rows have no
supervised loss but remain in context and draws. The second loss matches the
signed all/easy score aggregated over the original selected actions, only when
that selected subset is wholly known and nonempty. Missing/empty selected
subsets do not become zero-risk labels. Reference targets enter that loss only,
not model inputs. Selected-query sums divide by their summed causal row scales.

The loss is0.5 all-action risk MSE plus0.5 selected-query signed-score MSE.
Both arms use identical source-balanced fitting-only moment RMS, floor1e-6,
not the historical all-action0.05 floor. This shared change is explicit and
is not separately identified by the paired context comparison. Original causal
feature normalization is unchanged. Trace component gradient norms, alignment,
selected-query loss activation and unknown supervision during actual updates.

Six original contexts, three head seeds17/29/43,36 fits,2,000 updates per fit,
query batches16 equally allocated across four sources. Width64, AdamW lr0.001,
clip5, no model/threshold/checkpoint selection from outcomes. The forecaster
remains the single frozen producer seed43. Three critic seeds are not three
forecaster seeds or independent scenes.

## Decoding and full-policy evaluation

For each critic, recompute independent selection using the unchanged2% all/easy
rule. Optimize joint allocation at that critic's independent per-query count,
using the same inherited exact binary solver,2s/query limit and raw postchecks.
The pointwise solution is the feasible fallback. Benefit is frozen, but changed
harm also changes net predicted gain. No ground-truth future chooses an action.

Additional explicit matched controls use the smaller of the two independent
counts in EVERY query. Each trims its own feasible pointwise anchor by predicted
gain, then reoptimizes. These controls are utility-only, not deployment repairs.
Reuse an own-count optimum if the requested matched count is identical. Solver
fallbacks remain reported as nonoptimal; no numerical tolerance is relaxed.

Primary contrast: selected-set joint minus row-local joint full-policy paired
completion-lower utility. Report its known contrast and the explicit common-
count controls. Also compare both with the original independent/joint policies,
whose prediction hashes and144 full-policy scores must reproduce exactly.
No-intervention floor stays visible. Keep original complete-recording all/easy
positive-harm/reference2% and easy-degradation upper checks. Empty selected
support and empty matched comparisons are not estimable, not safe successes.

Report available-label ADE, full-roster FDE status, regret, hard/easy slices,
tails, intervention counts and predicted-path pair geometry proxies. Unknown
outcomes stay in completion bounds; partial ADE gaps are not silently certified.
Use3,000 locality bootstrap draws after context/seed averaging over12 exposed
localities. Nominal fitting intervals do not adjust for the preceding adaptive
development and do not constitute independent confirmation.

Fitting advancement needs the primary positive lower CI, positive floor lower
utility, unchanged complete-recording risk, exact matched design/controls and
reported main-query optimality. More accurate MSE or fewer switches alone cannot
pass. Even a fitting pass cannot promote independent or deployment status.

## Runtime and freeze

Use native arm64 .venv-pytorch CPU4/interop1/workers0. Local prior readouts use
approximately10GB and fit48GB; no HPC duplication is warranted before the real
training pilot. Checkpoint every100 updates, heartbeat every50. Permit healthy
runs up to12h, reserve10GiB disk and cap new private checkpoints at256MiB.
Do not materialize new raw, history, feature or latent stores.

Real pilot: first context/seed17,50 then100 updates per arm, compared with a
direct100-update repeat:400 actual updates. Require exact scientific recovery,
matched parameters/initialization/draws and nonzero real updates. Then complete
all36 fits; refit both first context/seed17 heads for4,000 additional updates.
Freeze all weights before the72-group policy readout and perform a separate
full inference/allocation/metric replay. Do not tune after readout.

Publish code/config/aggregate metrics/receipts/reports only. Keep data, normalized
features, row actions and checkpoints private. Retain unrelated staged work.
Task is obs8/pred12 at raw stride12, not supplemental raw-frame t+50. Detector-
silver image coordinates are not metric, seconds-level, human gold, true3D or
foundation evidence. No Stage5C execution or SMC.
