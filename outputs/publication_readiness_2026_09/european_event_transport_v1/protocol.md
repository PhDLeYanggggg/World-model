# Common-Event Residual Transport

## Material Passport

Source-development, one-factor follow-up to verified commit c621fa0a. Parent
432 native-Torch heads and 864 probes are cached_verified, not retrained here.
The OOF repair failed: 0/6 full-input positive MSE intervals vs original.
Inner/outer easy-event disagreement reaches 15.012%; it is a hypothesis,
not a demonstrated cause. This experiment runs 864 new fixed ridge fits.

## Hypothesis And Isolation

Does aligning the residual response to the outer-fitting event repair its
transport? Freeze producers, original outer heads, features, fitting weights,
bins, ridge 0.1, seeds, rosters, sample budget and evaluation definition.
For all 144 views, three banks and two probe arms, change only the supervised
residual target to H * E_outer_fit instead of H * E_inner_producer.
Keep every full/motion-only view and both cyclic controls.

The common cut is the already frozen positive-CV-ADE 25th percentile from the
three meta-fitting localities. It never uses the outer locality. Those three
localities legitimately provide meta-level supervision; the common cut must
not feed back into inner teacher training, preprocessing or predictions.
This is NOT a claim that the common target was independent of each inner-held
row. OOF describes base prediction provenance only. Frozen teachers still
predict their original inner event, which remains a transport limitation.
Future errors are supervision only, never inference features. All original
outer target hashes and D/H/D_E outputs must stay identical.

For the fixed ridge design, verify the exact pre-clipping decomposition:
old_shift - common_shift = ridge_projection(H * (E_inner - E_outer_fit)).
Report clipping separately; post-clip terms are not additive causal shares.
No outcome-dependent optimal shrinkage or threshold selection is permitted.

## Sequence And Fixed Criteria

Commit registration and code before fitting. Hash-check all parent public
artifacts/source bindings and private predictions. Fit all 864 probes, keep
resume receipts and heartbeat. Freeze and commit all outer predictions before
new readout. Previously exposed source-development outcomes are NOT restored
as independent test data by this sequence. Independent selection, calibration
and confirmation stay unopened.

Primary endpoint: easy-harm MSE on known positive-disagreement rows, unchanged.
Mechanism signal: six positive full-input paired MSE intervals versus old OOF
context. Overall repair signal additionally requires six positive intervals
against original, prior three-locality context, common global, common cyclic
next and common cyclic previous. Require no negative/missing top10 or coverage
intervals in those comparisons. Failure cannot be replaced by a secondary win.
This conservative source-development gate is not a safety/noninferiority test.
Three seeds are averaged per locality, then 3000 paired resamples of four
localities, same seed 38113. Six assignments overlap; no multiplicity-adjusted
claim, no window-independent CI. All comparisons and both input pairs retained.

Replay 864 probe fits/predictions, all 36 readout groups and independent MSE
arithmetic. No new neural optimization, trajectory training or intervention
policy evaluation. Even passing these diagnostic gates does not deploy a model.

## Resources And Boundaries

Native arm64 Python, CPU4/interop1/workers0; no multiprocessing/resource probing.
Local is suitable for these fixed low-dimensional fits. Maintain 10GiB reserve.
CREATE queue check is read-only; unrelated jobs untouched. Per-row files and
checkpoints remain private. Eight observed/twelve predicted annotation steps,
detector-derived image pixels; no metric, seconds, human gold, physical safety,
true3D, foundation or submission-readiness claim. Stage5C and SMC remain off.
