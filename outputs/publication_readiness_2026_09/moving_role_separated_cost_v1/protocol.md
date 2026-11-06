# Role-Separated Cost Learning And Full-Action Joint Selection

## Why This Is Not Another Generic Hurdle Repeat

The earlier observation-support SOURCE experiment already composes an observed
conditional cost with an unknown envelope. It loses observed utility and fails
the full risk contract; its identification arm retains a composed-bound anchor
loss and shared representation. That negative is preserved, not renamed.
The latest complete-risk donor experiment also worsens observed-target error
in all 24 source/seed views after unknown bounds are added to shared regression.

This comparison tests a narrower, new model-side change: **remove the composed
bound anchor from observed-cost learning, use independent conditional-cost and
label-support networks, and supervise their respective roles separately**.
Both new arms have exactly the same architecture, input normalization, known-only
moment RMS, initial weights, source/query draws, budget and five-moment decoder.
The control fits composed moments on the complete pool. The treatment fits
conditional moments only on observed rows plus proper unweighted support BCE
on the complete pool. No unknown-target magnitude can backpropagate into the
treatment's conditional-cost network. Support BCE does not touch that network.
The decomposition is not a novelty claim or proof of physical visibility.

## Why Candidate Coverage Must Change Too

Mean and unanimity have 6 and 12 recording views without selected easy reference.
Any policy that only vetoes their choices cannot create reference mass in those
views, even with perfect knowledge of wholly missing labels. Thus another
veto-only missingness classifier cannot pass the retained support contract.

Use the existing floor plus six fixed neural-amplitude actions, including
past-eligible rows on which the frozen proposer falls back. No new tiny/epsilon
action is introduced. Compare proposal-only, full-bank independent, uniform
frame, identical-count joint and free-count joint decisions using the same
new predictions. The existing exact/postchecked frame allocator maximizes
predicted gain under the unchanged two 2% moment budgets. Empty allocation
remains NA in the original evaluator, not a pass. Joint risk pooling is not
pairwise collision optimization or a physical-safety guarantee.

## Data, Inputs And Targets

Only source-held producer-A donors and the eight already-exposed own-B sources.
No new C/H, reserved Moving20, DUT or DroneCrowd access. The original 62,466-row
own-B population is retained in evaluation; unknown and nonproposal rows stay.
Training uses all legal amplitude candidates, with 50/50 fitting own-B/A mass
and source/query-balanced weights. The entire query source is excluded from
training and all learned normalization. Parent forecast ancestry stays fixed.

The conditional network receives the original past-only 120 features. A separate
label-support network receives their first 50 action-invariant agent/neighbor
motion features. Actual future validity is a training/evaluation label, never
an input or a query filter. No future endpoints, central velocity, eventual
track length, test endpoints/goals or test normalization. Targets B/H/R/ER/EH
follow the original available-label and unknown-envelope semantics. Missing
physical outcomes are not imputed and no missing-at-random assumption is made.

Use the same normalized causal envelope in composition. B/H stay bounded by
path disagreement; B<=R, ER<=R and EH<=H. Output probabilities describe released
label availability, not latent future visibility or baseline failure.

## Fixed Execution And Readout

48 real Torch fits: 2 arms x 8 source-excluded folds x 3 risk seeds. Each has
4,096 updates with a 64/32 conditional MLP and independent width32 support MLP.
All 48 complete exact refits finish before readout. An actual 128+128 versus
256-step pilot verifies recovery on this data, not only on synthetic tensors.
Final checkpoints only; no model/threshold selection on development outcomes.

Freeze complete causal predictions and choices before labels. Per-query solver
limit is 1 s; timeout/numerical failures retain the feasible independent parent,
are recorded and never called exact optima. Allocation checkpoints resume the
same frozen predictions. Replay recomputes predictions, verifies cached choices
and metrics, but does not rerun timed integer search or fit models.

Primary contrast: role-separated minus composed-anchor **joint-free** paired
completion-lower gain, averaged over three risk seeds then eight source blocks.
Use 3,000 paired source-component bootstrap resamples, not overlapping windows.
Also report known-label contrast, all five policy controls, per-source and
per-seed results, observed/bound errors, support-probability quality, intervention
counts and every original recording-risk/support result. Matched counts measure
utility only; empty matched utility remains not estimable.

A complete development safety candidate requires all original 24 recording
views per risk seed to pass and positive paired evidence; a smaller supported
subset or favorable seed does not replace it. No automatic deployment or new
independent test follows. This also changes model family and training candidate
coverage versus older two-channel heads; those comparisons are not isolated
loss effects. Only the two new arms and their fixed decision controls isolate
the stated factors.

## Runtime And Claims

Native arm64 Torch CPU4/inter-op1/workers0; checkpoint every128 updates, heartbeat
every256, explicit resume, 12-hour phase limit, 5 GiB free reserve, 2 GiB new
private-artifact cap. Pilot actual cost before full training. Use local resources
while they suffice; do not duplicate CREATE jobs or treat old scheduler records
as current job evidence. No stopping merely because training is slow.

This is exposed-development native obs8/pred12 image-coordinate silver evidence,
not independent confirmation, full12 ADE, metric/seconds, true3D or foundation.
Annotation-generation causality is unverified. Original24-group main remains NA;
old failures remain. Stage5C/SMC, identity certification and deployment stay false.
