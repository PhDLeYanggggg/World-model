# Track/Event Sampling Contrast

Registered before real fits. This is an exploratory repair experiment on the
already approved ETH, Hotel and grouped Zara fit cohort, not a new test or
admission of SDD as auxiliary training data. Students development, calibration
and confirmation remain closed. Eight observations, twelve native annotation
steps and equal-physical-scene past-normalized ADE are unchanged.

## Hypothesis and Fixed Comparison

Long tracks contribute many overlapping windows. Uniform row sampling may
underexpose scarce motion changes; balancing recording-local tracks, and then
supervised event categories within each training scene, may improve forecasting.
This does not create independent events or solve missing predictive information.

Four samplers: row-uniform, scene-uniform, scene/track-uniform, and
scene/event/track-uniform. Each hierarchical level distributes equal mass among
present groups, followed by uniform rows within a group. No missing event class
is synthesized. Event categories are exact-static/stays, exact-static/moves,
moving/stops, moving/turns and remaining motion. Static uses the registered
zero past displacement definition; stop uses four equal final target positions;
turn uses at least 45 degrees between past and future final-four displacement.
Stop takes precedence over turn. These supervised proxies are not human labels.

Categories use training targets only and never enter model inputs or inference
gates. Held-fit targets are used after the fixed model is frozen for descriptive
evaluation slices only. Track identity means recording plus source agent ID,
not a certified unique person across recordings. Models see neither ID nor event.

Keep the observed-motion-v2 MLP, CV skip, log-ADE loss, optimizer and 4,000-update
budget unchanged. Compare quality-control and directed observed-motion features
at seeds 17/29/43 and all three leave-physical-scene-out fit folds. Reuse 18 old
row-log controls only after full dependency hashes and exact checkpoint inference
replay. Train 54 new fits, 216,000 updates. Report source separately.

## Verification and Decision Rule

First verify synthetic sampler masses, held-target mutation isolation, old/new
row-training equivalence and exact resume. Pilot one registered fresh fit to 200
then 400 updates without held evaluation; resume it to the fixed 4,000 endpoint.
Checkpoint every 400 updates, CPU threads 4, interop 1, no DataLoader processes.
All conditions are reported; no best checkpoint or sampler is selected using
held-fit performance. Retain failure slices and easy absolute harm, including
undefined percentage degradation when reference error is zero.

Report paired three-scene exploratory bootstrap (2,000 draws), per-seed and
per-scene errors, event/source-track support, expected and actual training draw
counts, training-versus-held gap and tail error. A useful next-stage signal needs
positive primary gain, easy degradation no greater than 2%, and evidence the
effect is not confined to one scene. This is not an independent promotion gate.
If only training/event subsets improve but held primary/easy do not, sampling
alone has not repaired the model. Do not open sealed roles to chase improvement.

No correction deployment, new data admission, physical-seconds/metric claim,
latent generation or SMC. Offline annotation lineage is disclosed; no strict
sensor-as-of claim. Preserve old controls and all failed results.
